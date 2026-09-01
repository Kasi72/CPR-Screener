"""
download_bhavcopy.py -- Phase C: NSE Equity Bhavcopy downloader

Downloads NSE CM bhavcopy CSVs for a date range and extracts DELIV_PER
(delivery %) per symbol per day.  Saves to models/delivery_data.pkl.

Usage:
    python scripts/ml/download_bhavcopy.py
    python scripts/ml/download_bhavcopy.py --start 2019-01-01 --end 2024-12-31

NSE provides bhavcopy in two formats depending on date:
  New (post-2019): sec_bhavdata_full_{DDMMYYYY}.csv
  Old (pre-2019):  cm{DD}{MON}{YYYY}bhav.csv.zip

Requires 'requests' (already in requirements).
Downloads can take 15-30 min for 5 years; re-run resumes from last saved date.
"""

import os, sys, pickle, time, zipfile, io, argparse
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.ml.data_utils import MODELS_DIR
CACHE_FILE  = os.path.join(MODELS_DIR, "delivery_data.pkl")
MONTH_ABBR  = ['JAN','FEB','MAR','APR','MAY','JUN',
               'JUL','AUG','SEP','OCT','NOV','DEC']

HEADERS = {
    'User-Agent':      ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                        'AppleWebKit/537.36 (KHTML, like Gecko) '
                        'Chrome/120.0.0.0 Safari/537.36'),
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept':          'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection':      'keep-alive',
    'Referer':         'https://www.nseindia.com/',
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)
_playwright_tried = False


def _init_session_playwright():
    """
    Use Playwright stealth browser to acquire real NSE cookies,
    then inject them into the requests SESSION for bulk downloads.
    Returns True on success.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False

    try:
        print("Launching stealth browser to acquire NSE cookies...")
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context(
                user_agent=(
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/120.0.0.0 Safari/537.36'
                ),
                locale='en-IN',
                timezone_id='Asia/Kolkata',
            )
            page = ctx.new_page()
            # Step 1: load NSE main site (sets main domain cookies)
            page.goto('https://www.nseindia.com', wait_until='networkidle', timeout=30000)
            page.wait_for_timeout(3000)
            # Step 2: load archives root (sets nsearchives cookies)
            page.goto('https://nsearchives.nseindia.com', timeout=20000)
            page.wait_for_timeout(2000)
            cookies = ctx.cookies()
            browser.close()

        # Inject all cookies into requests Session
        for c in cookies:
            SESSION.cookies.set(c['name'], c['value'], domain=c.get('domain', ''))
        print(f"Stealth session ready: {len(cookies)} cookies acquired.")
        return True
    except Exception as e:
        print(f"WARN: Playwright session failed ({e})")
        return False


def _init_session():
    """On 403: attempt Playwright cookie refresh once per run."""
    global _playwright_tried
    if not _playwright_tried:
        _playwright_tried = True
        _init_session_playwright()


def _fetch_new_format(dt, verbose=False):
    """POST-2021 format: sec_bhavdata_full_DDMMYYYY.csv"""
    dstr = dt.strftime('%d%m%Y')
    urls = [
        # archives.nseindia.com confirmed working without cookies (nsearchives has HTTP2 issues)
        f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{dstr}.csv",
        f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{dstr}.csv",
    ]
    for url in urls:
        try:
            r = SESSION.get(url, timeout=30)
            if verbose:
                print(f"  DEBUG new_fmt {dt.date()} {url.split('//')[1].split('/')[0]} -> HTTP {r.status_code} len={len(r.content)}")
            if r.status_code == 200 and len(r.content) > 1000:
                return r.content
            if r.status_code == 403:
                _init_session()
                time.sleep(3)
        except Exception as e:
            if verbose:
                print(f"  DEBUG new_fmt {dt.date()} -> EXC {e}")
    return None


def _fetch_old_format(dt, verbose=False):
    """PRE-2021 format: cmDDMONYYYYbhav.csv.zip — try both NSE domains."""
    mon  = MONTH_ABBR[dt.month - 1]
    dstr = dt.strftime('%d') + mon + dt.strftime('%Y')
    urls = [
        f"https://archives.nseindia.com/archives/equities/bhavcopy/cm{dstr}bhav.csv.zip",
        f"https://nsearchives.nseindia.com/archives/equities/bhavcopy/cm{dstr}bhav.csv.zip",
    ]
    for url in urls:
        for attempt in range(2):
            try:
                r = SESSION.get(url, timeout=30)
                if verbose:
                    print(f"  DEBUG old_fmt {dt.date()} {url.split('/')[-1]} -> HTTP {r.status_code} len={len(r.content)}")
                if r.status_code == 200 and len(r.content) > 500:
                    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                        name = zf.namelist()[0]
                        return zf.read(name)
                if r.status_code == 403 and attempt == 0:
                    _init_session()
                    time.sleep(3)
            except Exception as e:
                if verbose:
                    print(f"  DEBUG old_fmt {dt.date()} -> EXC {e}")
                if attempt == 0:
                    time.sleep(2)
    return None


def _parse_bhavcopy(raw_bytes, dt):
    """Parse raw CSV bytes -> DataFrame with columns: SYMBOL, DELIV_PER, DATE"""
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
        df.columns = [c.strip().upper().replace(' ', '_') for c in df.columns]

        # New format has DELIVERY_QTY + TTL_TRD_QNTY; compute pct
        if 'DELIV_PER' in df.columns:
            pct_col = 'DELIV_PER'
        elif 'DELIVERY_QTY' in df.columns and 'TTL_TRD_QNTY' in df.columns:
            df['DELIV_PER'] = (df['DELIVERY_QTY'] /
                               df['TTL_TRD_QNTY'].replace(0, np.nan) * 100).fillna(0)
            pct_col = 'DELIV_PER'
        else:
            return None

        sym_col = next((c for c in ['SYMBOL', 'SYM'] if c in df.columns), None)
        if sym_col is None:
            return None

        out = df[[sym_col, pct_col]].copy()
        out.columns = ['symbol', 'deliv_pct']
        out['date'] = dt.date()
        out['deliv_pct'] = pd.to_numeric(out['deliv_pct'], errors='coerce').fillna(0)
        return out[out['deliv_pct'] > 0]
    except Exception:
        return None


def download_bhavcopy(start='2019-01-01', end=None, delay=1.0):
    """
    Download and parse bhavcopy for all trading days in [start, end].
    Resumes from last saved state. Saves to CACHE_FILE.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    _init_session()

    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')

    # Load existing cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            existing = pickle.load(f)
        if len(existing) > 0 and not existing['date'].isna().all():
            print(f"Loaded existing cache: {len(existing)} rows, "
                  f"dates {existing['date'].min()} to {existing['date'].max()}")
            last_date = pd.Timestamp(existing['date'].max())
            start_dt  = last_date + pd.Timedelta(days=1)
        else:
            print("Cache file empty — starting fresh.")
            existing = pd.DataFrame(columns=['symbol', 'deliv_pct', 'date'])
            start_dt = pd.Timestamp(start)
    else:
        existing  = pd.DataFrame(columns=['symbol', 'deliv_pct', 'date'])
        start_dt  = pd.Timestamp(start)

    end_dt    = pd.Timestamp(end)
    all_dates = pd.bdate_range(start_dt, end_dt)  # business days only

    if len(all_dates) == 0:
        print("Cache is up to date.")
        return existing

    print(f"Downloading {len(all_dates)} trading days "
          f"({start_dt.date()} to {end_dt.date()})...")

    chunks = [existing]
    ok_count = fail_count = 0

    for i, dt in enumerate(all_dates):
        verbose = (i < 3)   # print debug for first 3 days only
        raw = _fetch_new_format(dt, verbose=verbose) or _fetch_old_format(dt, verbose=verbose)
        if raw is None:
            fail_count += 1
        else:
            parsed = _parse_bhavcopy(raw, dt)
            if parsed is not None and not parsed.empty:
                chunks.append(parsed)
                ok_count += 1
            else:
                fail_count += 1

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(all_dates)}  ok={ok_count}  fail={fail_count}")
            # Save incrementally every 20 days
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


def load_delivery_data():
    """
    Load cached delivery data. Returns pivot DataFrame:
    index=date, columns=symbol, values=deliv_pct.
    Returns None if cache missing.
    """
    if not os.path.exists(CACHE_FILE):
        return None
    with open(CACHE_FILE, 'rb') as f:
        df = pickle.load(f)
    # Pivot for fast lookup: df.loc[date, symbol]
    df['date'] = pd.to_datetime(df['date'])
    pivot = df.pivot_table(index='date', columns='symbol',
                           values='deliv_pct', aggfunc='first')
    print(f"[delivery] Loaded {len(pivot)} dates x {len(pivot.columns)} symbols")
    return pivot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2019-01-01')
    parser.add_argument('--end',   default=None)
    parser.add_argument('--delay', type=float, default=1.0,
                        help='seconds between requests (default 1.0)')
    args = parser.parse_args()
    download_bhavcopy(start=args.start, end=args.end, delay=args.delay)
