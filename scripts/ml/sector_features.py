"""
sector_features.py -- Phase B: Market / Sector Relative Strength features

Downloads Nifty 50 + major NSE sector index daily closes once, caches to
models/sector_data.pkl.  Exposes:
    download_sector_data()       -- run once to populate cache
    load_sector_closes()         -- returns dict {ticker: Series(date->close)}
    compute_rs(sym_closes, dates_arr, signal_idx, bench_closes, period)
                                 -- relative-strength ratio at a point in time
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.ml.data_utils import MODELS_DIR
CACHE_FILE = os.path.join(MODELS_DIR, "sector_data.pkl")

# Sector indices available on yfinance for NSE
# Grouped so we can fall back gracefully if a ticker fails
SECTOR_TICKERS = {
    '^NSEI':      'Nifty50',       # market benchmark (always downloaded)
    '^CNXIT':     'IT',
    '^NSEBANK':   'Bank',
    '^CNXPHARMA': 'Pharma',
    '^CNXAUTO':   'Auto',
    '^CNXFMCG':   'FMCG',
    '^CNXMETAL':  'Metal',
    '^CNXREALTY': 'Realty',
    '^CNXENERGY': 'Energy',
    '^CNXMEDIA':  'Media',
}

# yfinance sector string -> NSE sector ticker
YFINANCE_SECTOR_MAP = {
    'Technology':             '^CNXIT',
    'Communication Services': '^CNXMEDIA',
    'Healthcare':             '^CNXPHARMA',
    'Consumer Defensive':     '^CNXFMCG',
    'Consumer Cyclical':      '^CNXAUTO',
    'Financial Services':     '^NSEBANK',
    'Basic Materials':        '^CNXMETAL',
    'Energy':                 '^CNXENERGY',
    'Real Estate':            '^CNXREALTY',
    'Industrials':            '^NSEI',
    'Utilities':              '^NSEI',
}


def download_sector_data(start='2019-01-01', end=None):
    """
    Download all sector index daily closes and save to cache.
    Run once before build_dataset.py.
    """
    import yfinance as yf
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')

    os.makedirs(MODELS_DIR, exist_ok=True)
    sector_closes = {}

    print("Downloading NSE sector index data...")
    for ticker, name in SECTOR_TICKERS.items():
        try:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True)
            if df.empty:
                print(f"  SKIP {ticker} ({name}): no data")
                continue
            # Flatten MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            closes = df['Close'].dropna()
            closes.index = pd.to_datetime(closes.index).normalize()
            sector_closes[ticker] = closes
            print(f"  OK   {ticker} ({name}): {len(closes)} rows "
                  f"{closes.index[0].date()} -> {closes.index[-1].date()}")
        except Exception as e:
            print(f"  FAIL {ticker} ({name}): {e}")

    if '^NSEI' not in sector_closes:
        raise RuntimeError("Nifty 50 (^NSEI) download failed -- cannot proceed.")

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(sector_closes, f)
    print(f"\nSaved sector data ({len(sector_closes)} indices) -> {CACHE_FILE}")
    return sector_closes


def load_sector_closes():
    """Load cached sector closes dict. Returns {} if cache missing."""
    if not os.path.exists(CACHE_FILE):
        return {}
    with open(CACHE_FILE, 'rb') as f:
        return pickle.load(f)


def _rs_at(sym_closes, sig_idx, bench_ser, period):
    """
    Relative strength of stock vs benchmark over `period` bars ending at sig_idx.
    Returns ratio: 1.0 = in-line, >1 = outperforming, <1 = lagging.
    """
    if sig_idx < period:
        return 1.0
    sig_date = sym_closes.index[sig_idx]
    sig_date_past = sym_closes.index[max(0, sig_idx - period)]

    c_now  = float(sym_closes.iloc[sig_idx])
    c_past = float(sym_closes.iloc[sig_idx - period])
    if c_past <= 0:
        return 1.0
    stock_ret = c_now / c_past - 1.0

    # Find matching bench dates
    bench_now  = bench_ser.asof(sig_date)
    bench_past = bench_ser.asof(sig_date_past)
    if bench_past <= 0 or pd.isna(bench_now) or pd.isna(bench_past):
        return 1.0
    bench_ret = float(bench_now) / float(bench_past) - 1.0

    # Ratio: stock return / bench return (clipped to avoid blowups)
    if abs(bench_ret) < 0.0001:
        return 1.0 + np.clip(stock_ret, -5.0, 5.0)
    return np.clip(1.0 + (stock_ret - bench_ret), -3.0, 5.0)


def get_rs_features(sym_closes_series, signal_idx, sector_closes,
                    sector_ticker='^NSEI'):
    """
    Compute market_rs_5d, market_rs_20d, sector_rs_5d, sector_rs_20d.
    Falls back to ^NSEI if sector_ticker not in cache.
    """
    bench_nsei   = sector_closes.get('^NSEI')
    bench_sector = sector_closes.get(sector_ticker, bench_nsei)

    if bench_nsei is None:
        return 1.0, 1.0, 1.0, 1.0

    market_rs_5d  = _rs_at(sym_closes_series, signal_idx, bench_nsei,   5)
    market_rs_20d = _rs_at(sym_closes_series, signal_idx, bench_nsei,  20)
    sector_rs_5d  = _rs_at(sym_closes_series, signal_idx, bench_sector,  5)
    sector_rs_20d = _rs_at(sym_closes_series, signal_idx, bench_sector, 20)

    return float(market_rs_5d), float(market_rs_20d), \
           float(sector_rs_5d), float(sector_rs_20d)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2019-01-01')
    parser.add_argument('--end',   default=None)
    args = parser.parse_args()
    download_sector_data(start=args.start, end=args.end)
