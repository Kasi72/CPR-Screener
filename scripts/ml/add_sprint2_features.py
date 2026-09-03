"""
Sprint 2 feature augmentation — pure streaming, minimal RAM.

Loads OHLCV dict (~150MB), then streams signal_dataset.csv one row at a time
via Python csv module. Never loads full signal CSV into memory.

New columns:
  open_inside_cpr, cpr_virgin, consecutive_narrow_cprs,
  cpr_midpoint_trend, cpr_expansion_factor
"""

import csv, os, sys, shutil
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.ml.data_utils import DATA_FILE, MODELS_DIR, calc_cpr

SIGNAL_CSV = os.path.join(MODELS_DIR, 'signal_dataset.csv')
TMP_CSV    = SIGNAL_CSV + '.sprint2_tmp'
NEW_COLS   = ['open_inside_cpr', 'cpr_virgin', 'consecutive_narrow_cprs',
              'cpr_midpoint_trend', 'cpr_expansion_factor']


def load_ohlcv():
    """Load OHLCV into per-symbol numpy arrays — ~150MB total."""
    import pandas as pd
    print(f"Loading OHLCV from {DATA_FILE} ...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df.columns = [c.strip().title() for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(['Symbol', 'Date'])

    ohlcv_by_sym = {}
    for sym, g in df.groupby('Symbol', sort=False):
        g = g.reset_index(drop=True)
        ohlcv_by_sym[sym] = {
            'dates':  g['Date'].values.astype('datetime64[ns]'),
            'opens':  g['Open'].values.astype(np.float64),
            'highs':  g['High'].values.astype(np.float64),
            'lows':   g['Low'].values.astype(np.float64),
            'closes': g['Close'].values.astype(np.float64),
        }
    n = len(ohlcv_by_sym)
    print(f"  {n:,} symbols loaded")
    return ohlcv_by_sym


def compute_features(sym_data, sig_ts):
    """Compute 5 Sprint 2 features for one signal row."""
    dates  = sym_data['dates']
    opens  = sym_data['opens']
    highs  = sym_data['highs']
    lows   = sym_data['lows']
    closes = sym_data['closes']

    bar_i = int(np.searchsorted(dates, sig_ts, side='left'))
    if bar_i < 1 or bar_i >= len(dates):
        return [0, 0, 0.0, 0.0, 1.0]

    i   = bar_i
    cpr = calc_cpr(highs[i-1], lows[i-1], closes[i-1])

    # ATR
    lo      = max(0, i - 14)
    cur_atr = float(np.mean(highs[lo:i] - lows[lo:i])) or 0.001

    # 1. open_inside_cpr
    open_inside = 1 if cpr['lower'] <= opens[i] <= cpr['upper'] else 0

    # 2. cpr_virgin
    virgin = 1
    for _j in range(max(0, i - 20), i):
        if lows[_j] <= cpr['upper'] and highs[_j] >= cpr['lower']:
            virgin = 0
            break

    # Shared lookback
    lb_start = max(1, i - 19)
    widths   = []
    mids     = []
    for _j in range(lb_start, i + 1):
        _c = calc_cpr(highs[_j-1], lows[_j-1], closes[_j-1])
        widths.append(_c['width_pct'])
        mids.append((_c['upper'] + _c['lower']) / 2.0)

    # 3. consecutive_narrow_cprs
    hist_w = widths[:-1]
    med_w  = float(np.median(hist_w)) if hist_w else widths[-1]
    consec = 0
    for _w in reversed(hist_w):
        if _w < med_w:
            consec += 1
        else:
            break
    consecutive = float(min(consec, 10))

    # 4. cpr_midpoint_trend
    hist_mids = mids[-6:-1]
    if len(hist_mids) >= 2:
        x     = np.arange(len(hist_mids), dtype=np.float32)
        slope = float(np.polyfit(x, hist_mids, 1)[0])
        trend = float(np.clip(slope / cur_atr, -5.0, 5.0))
    else:
        trend = 0.0

    # 5. cpr_expansion_factor
    if len(widths) >= 2:
        expansion = float(np.clip(widths[-1] / max(widths[-2], 0.001), 0.1, 10.0))
    else:
        expansion = 1.0

    return [open_inside, virgin, consecutive, round(trend, 4), round(expansion, 4)]


def main():
    ohlcv = load_ohlcv()

    print(f"\nStreaming {SIGNAL_CSV} -> {TMP_CSV} ...")
    n_done = 0
    n_miss = 0

    with open(SIGNAL_CSV, 'r', newline='', encoding='utf-8') as fin, \
         open(TMP_CSV,    'w', newline='', encoding='utf-8') as fout:

        reader = csv.DictReader(fin)
        # Output header = all original cols + 5 new ones (drop old if exist)
        orig_cols = [c for c in reader.fieldnames if c not in NEW_COLS]
        out_fields = orig_cols + NEW_COLS
        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()

        for row in reader:
            sym    = row['symbol']
            date_s = row['date']

            if sym in ohlcv:
                try:
                    sig_ts = np.datetime64(date_s, 'ns')
                    feats  = compute_features(ohlcv[sym], sig_ts)
                except Exception:
                    feats = [0, 0, 0.0, 0.0, 1.0]
                    n_miss += 1
            else:
                feats = [0, 0, 0.0, 0.0, 1.0]
                n_miss += 1

            for col, val in zip(NEW_COLS, feats):
                row[col] = val
            # Only write orig_cols + new cols (drops stale new cols if present)
            writer.writerow({k: row.get(k, '') for k in out_fields})

            n_done += 1
            if n_done % 100_000 == 0:
                print(f"  {n_done:,} rows processed ...", flush=True)

    print(f"\n{n_done:,} rows done, {n_miss:,} symbol misses")
    print(f"Replacing {SIGNAL_CSV} ...")
    shutil.move(TMP_CSV, SIGNAL_CSV)
    print("Done.")


if __name__ == '__main__':
    main()
