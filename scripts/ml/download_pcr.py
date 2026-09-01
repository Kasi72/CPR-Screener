"""
download_pcr.py -- Phase D: NSE Options Put-Call Ratio per symbol

Downloads NSE FO bhavcopy ZIPs for a date range, extracts per-symbol
PCR = PE_OI / CE_OI (stock options only). Saves to models/pcr_data.pkl.

Usage:
    python scripts/ml/download_pcr.py
    python scripts/ml/download_pcr.py --start 2022-01-01 --end 2024-12-31

NSE FO bhavcopy URL:
  https://archives.nseindia.com/content/historical/DERIVATIVES/{YEAR}/{MON}/fo{DD}{MON}{YYYY}bhav.csv.zip

PCR interpretation:
  PCR > 1.2  = heavy put hedging → bearish institutional positioning
  PCR < 0.7  = heavy call buying  → bullish but may signal complacency
  PCR ~ 1.0  = neutral
"""

import os, sys, pickle, time, zipfile, io, argparse
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.ml.data_utils import MODELS_DIR
CACHE_FILE = os.path.join(MODELS_DIR, "pcr_data.pkl")
MONTH_ABBR = ['JAN','FEB','MAR','APR','MAY','JUN',
               'JUL','AUG','SEP','OCT','NOV','DEC']

HEADERS = {
    'User-Agent':      ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                        'AppleWebKit/537.36 (KHTML, like Gecko) '
                        'Chrome/120.0.0.0 Safari/537.36'),
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept':          'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer':         'https://www.nseindia.com/',
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)
_playwright_tried = False


def _init_session_playwright():
    """Use Playwright stealth browser to acquire real NSE cookies."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("WARN: playwright not installed — skipping cookie acquisition.")
        return
    try:
        print("Launching stealth browser to acquire NSE cookies...")
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context(
                user_agent=HEADERS['User-Agent'],
                locale='en-IN',
                timezone_id='Asia/Kolkata',
            )
            page = ctx.new_page()
            try:
                page.goto('https://www.nseindia.com', timeout=20000,
                          wait_until='domcontentloaded')
                page.wait_for_timeout(3000)
            except Exception:
                pass
            try:
                page.goto('https://archives.nseindia.com', timeout=15000,
                          wait_until='domcontentloaded')
                page.wait_for_timeout(2000)
            except Exception:
                pass
            cookies = ctx.cookies()
            browser.close()
        for c in cookies:
            SESSION.cookies.set(c['name'], c['value'], domain=c.get('domain', ''))
        print(f"Stealth session ready: {len(cookies)} cookies acquired.")
    except Exception as e:
        print(f"WARN: Playwright session failed ({e})")


def _init_session():
    """On 403: attempt Playwright cookie refresh once per run."""
    global _playwright_tried
    if not _playwright_tried:
        _playwright_tried = True
        _init_session_playwright()


def _fetch_fo_bhavcopy(dt, verbose=False):
    """
    Download FO bhavcopy for a given date.
    NSE uses two formats:
      Old (pre-~Oct 2024): fo{DD}{MON}{YYYY}bhav.csv.zip  (ZIP of CSV)
      New (post-~Oct 2024): BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.gz  (gzip CSV)
    Returns raw CSV bytes, or None.
    """
    import gzip
    mon    = MONTH_ABBR[dt.month - 1]
    year   = dt.strftime('%Y')
    dstr   = dt.strftime('%d') + mon + year
    ymd    = dt.strftime('%Y%m%d')

    # New format (gz) tried first for dates >= 2024 to avoid wasted ZIP attempts
    new_urls = [
        f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.gz",
        f"https://archives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.gz",
    ]
    # Old format (zip)
    old_urls = [
        f"https://archives.nseindia.com/content/historical/DERIVATIVES/{year}/{mon}/fo{dstr}bhav.csv.zip",
        f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{year}/{mon}/fo{dstr}bhav.csv.zip",
    ]

    # For newer dates try new format first; for old dates try old format first
    if dt >= pd.Timestamp('2024-09-01'):
        url_groups = [('gz', new_urls), ('zip', old_urls)]
    else:
        url_groups = [('zip', old_urls), ('gz', new_urls)]

    for fmt, urls in url_groups:
        for url in urls:
            try:
                r = SESSION.get(url, timeout=30)
                if verbose:
                    print(f"  DEBUG {dt.date()} [{fmt}] {url.split('/')[2]} -> HTTP {r.status_code} len={len(r.content)}")
                if r.status_code == 403:
                    _init_session()
                    r = SESSION.get(url, timeout=30)
                    if verbose:
                        print(f"  DEBUG {dt.date()} [{fmt}] retry -> HTTP {r.status_code} len={len(r.content)}")
                if r.status_code == 200 and len(r.content) > 500:
                    if fmt == 'zip':
                        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                            return zf.read(zf.namelist()[0])
                    else:  # gz
                        return gzip.decompress(r.content)
            except Exception as e:
                if verbose:
                    print(f"  DEBUG {dt.date()} [{fmt}] -> EXC {e}")
    return None


def _parse_fo_bhavcopy(raw_bytes, dt):
    """
    Parse FO bhavcopy CSV → per-symbol PCR DataFrame.
    Handles both old format (INSTRUMENT/SYMBOL/OPTION_TYP/OPEN_INT)
    and new format (FinInstrmTp/TckrSymb/OptnTp/SsnltyAmt).
    Returns DataFrame with columns: symbol, pcr, date.
    PCR = PE_OI / CE_OI (stock options only, clipped to [0, 10]).
    """
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
        df.columns = [c.strip() for c in df.columns]

        # Detect format by presence of canonical old columns
        if 'INSTRUMENT' in df.columns:
            # Old format
            df.columns = [c.upper().replace(' ', '_') for c in df.columns]
            instrument_col = 'INSTRUMENT'
            symbol_col     = 'SYMBOL'
            opttype_col    = 'OPTION_TYP'
            oi_col         = 'OPEN_INT'
            optstk_val     = 'OPTSTK'
            ce_val, pe_val = 'CE', 'PE'
        elif 'FinInstrmTp' in df.columns or 'TckrSymb' in df.columns:
            # New format (post Oct 2024) — instrument type col may be absent in some variants
            instrument_col = 'FinInstrmTp' if 'FinInstrmTp' in df.columns else None
            symbol_col     = 'TckrSymb'
            opttype_col    = 'OptnTp'
            # New format OI column: SsnltyAmt or OpnIntrst
            oi_col = next((c for c in ['OpnIntrst', 'SsnltyAmt'] if c in df.columns), None)
            if oi_col is None:
                return None
            optstk_val = 'OPTSTK'
            ce_val, pe_val = 'CE', 'PE'
        else:
            return None

        if instrument_col is not None:
            df = df[df[instrument_col] == optstk_val].copy()
        elif opttype_col in df.columns:
            # No instrument type column — scope to CE/PE rows only
            df = df[df[opttype_col].isin([ce_val, pe_val])].copy()
        else:
            return None
        if df.empty:
            return None

        df[oi_col] = pd.to_numeric(df[oi_col], errors='coerce').fillna(0)

        # Stock options (OPTSTK): per-symbol PCR
        grp = df.groupby([symbol_col, opttype_col])[oi_col].sum().unstack(fill_value=0)
        pe_oi = grp.get(pe_val, pd.Series(0, index=grp.index))
        ce_oi = grp.get(ce_val, pd.Series(0, index=grp.index))

        pcr = (pe_oi / ce_oi.replace(0, np.nan)).fillna(0).clip(0, 10)
        out = pd.DataFrame({'symbol': pcr.index, 'pcr': pcr.values})
        out['date'] = dt.date()
        result = out[out['pcr'] > 0].copy()

        # Index options (OPTIDX): market-wide Nifty PCR — extract separately
        try:
            if instrument_col is not None:
                df_full = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
                df_full.columns = [c.strip() for c in df_full.columns]
                if 'INSTRUMENT' in df_full.columns:
                    df_full.columns = [c.upper().replace(' ', '_') for c in df_full.columns]
                    df_idx = df_full[df_full['INSTRUMENT'] == 'OPTIDX'].copy()
                    sym_col_idx, opt_col_idx, oi_col_idx = 'SYMBOL', 'OPTION_TYP', 'OPEN_INT'
                elif 'FinInstrmTp' in df_full.columns:
                    df_idx = df_full[df_full.get('FinInstrmTp', pd.Series()).eq('OPTIDX')].copy() \
                             if 'FinInstrmTp' in df_full.columns else pd.DataFrame()
                    sym_col_idx, opt_col_idx = 'TckrSymb', 'OptnTp'
                    oi_col_idx = next((c for c in ['OpnIntrst', 'SsnltyAmt'] if c in df_full.columns), None)
                else:
                    df_idx = pd.DataFrame()
                    oi_col_idx = None

                if not df_idx.empty and oi_col_idx is not None:
                    nifty_idx = df_idx[df_idx[sym_col_idx].isin(['NIFTY', 'BANKNIFTY', 'FINNIFTY'])].copy()
                    nifty_idx[oi_col_idx] = pd.to_numeric(nifty_idx[oi_col_idx], errors='coerce').fillna(0)
                    nifty_grp = nifty_idx.groupby([sym_col_idx, opt_col_idx])[oi_col_idx].sum().unstack(fill_value=0)
                    nifty_pe = nifty_grp.get('PE', pd.Series(0, index=nifty_grp.index))
                    nifty_ce = nifty_grp.get('CE', pd.Series(0, index=nifty_grp.index))
                    nifty_pcr = (nifty_pe / nifty_ce.replace(0, np.nan)).fillna(0).clip(0, 10)
                    # Store as special symbol '__MARKET__' for market-wide PCR
                    mkt_rows = pd.DataFrame({
                        'symbol': [f'__MKT_{s}__' for s in nifty_pcr.index],
                        'pcr': nifty_pcr.values,
                        'date': dt.date(),
                    })
                    result = pd.concat([result, mkt_rows[mkt_rows['pcr'] > 0]], ignore_index=True)
        except Exception:
            pass

        return result
    except Exception:
        return None


def download_pcr(start='2022-01-01', end=None, delay=1.0):
    """
    Download and parse FO bhavcopy for all trading days in [start, end].
    Resumes from last saved state. Saves to CACHE_FILE.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')

    # Load existing cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            existing = pickle.load(f)
        if len(existing) > 0 and not existing['date'].isna().all():
            print(f"Loaded cache: {len(existing)} rows, "
                  f"dates {existing['date'].min()} to {existing['date'].max()}")
            last_date = pd.Timestamp(existing['date'].max())
            start_dt  = last_date + pd.Timedelta(days=1)
        else:
            existing  = pd.DataFrame(columns=['symbol', 'pcr', 'date'])
            start_dt  = pd.Timestamp(start)
    else:
        existing  = pd.DataFrame(columns=['symbol', 'pcr', 'date'])
        start_dt  = pd.Timestamp(start)

    end_dt    = pd.Timestamp(end)
    all_dates = pd.bdate_range(start_dt, end_dt)

    if len(all_dates) == 0:
        print("Cache is up to date.")
        return existing

    print(f"Downloading {len(all_dates)} trading days ({start_dt.date()} to {end_dt.date()})...")

    chunks    = [existing]
    ok_count  = fail_count = 0

    for i, dt in enumerate(all_dates):
        verbose = (i < 3)
        raw = _fetch_fo_bhavcopy(dt, verbose=verbose)
        if raw is None:
            fail_count += 1
        else:
            parsed = _parse_fo_bhavcopy(raw, dt)
            if parsed is not None and not parsed.empty:
                chunks.append(parsed)
                ok_count += 1
            else:
                fail_count += 1

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(all_dates)}  ok={ok_count}  fail={fail_count}")
            merged = pd.concat(chunks, ignore_index=True)
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(merged, f)
            chunks = [merged]

        time.sleep(delay)

    merged = pd.concat(chunks, ignore_index=True)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(merged, f)

    print(f"\nDone. {ok_count} days fetched, {fail_count} failed/skipped.")
    print(f"Total rows: {len(merged)}  saved -> {CACHE_FILE}")
    return merged


def load_pcr_data():
    """
    Load cached PCR data. Returns pivot DataFrame:
    index=date, columns=symbol, values=pcr.
    Returns None if cache missing.
    """
    if not os.path.exists(CACHE_FILE):
        return None
    with open(CACHE_FILE, 'rb') as f:
        df = pickle.load(f)
    df['date'] = pd.to_datetime(df['date'])
    # Exclude market-wide rows from per-symbol pivot
    df_sym = df[~df['symbol'].astype(str).str.startswith('__MKT_')]
    pivot = df_sym.pivot_table(index='date', columns='symbol', values='pcr', aggfunc='first')
    print(f"[pcr] Loaded {len(pivot)} dates x {len(pivot.columns)} symbols")
    return pivot


def load_market_pcr():
    """
    Load market-wide NIFTY PCR (index options). Returns a date-indexed Series.
    Falls back to None if no market PCR data available.
    """
    if not os.path.exists(CACHE_FILE):
        return None
    with open(CACHE_FILE, 'rb') as f:
        df = pickle.load(f)
    df['date'] = pd.to_datetime(df['date'])
    df_mkt = df[df['symbol'].astype(str).str.startswith('__MKT_')]
    if df_mkt.empty:
        return None
    # Aggregate across NIFTY/BANKNIFTY/FINNIFTY — prefer NIFTY, else mean
    nifty_rows = df_mkt[df_mkt['symbol'] == '__MKT_NIFTY__']
    if len(nifty_rows) > 0:
        series = nifty_rows.groupby('date')['pcr'].first()
    else:
        series = df_mkt.groupby('date')['pcr'].mean()
    print(f"[pcr] Market PCR: {len(series)} dates ({series.index.min().date()} to {series.index.max().date()})")
    return series


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2022-01-01')
    parser.add_argument('--end',   default=None)
    parser.add_argument('--delay', type=float, default=1.0)
    args = parser.parse_args()
    download_pcr(start=args.start, end=args.end, delay=args.delay)
