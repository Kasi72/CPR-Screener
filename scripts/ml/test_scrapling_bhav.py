"""
Quick test: use Scrapling StealthyFetcher to grab NSE bhavcopy.
Tries a few recent dates to find one that works.
"""
# [STANDALONE] One-off analysis script — not part of the ML training pipeline.

import sys
from scrapling import StealthyFetcher
import pandas as pd
import io

MONTH_ABBR = ['JAN','FEB','MAR','APR','MAY','JUN',
               'JUL','AUG','SEP','OCT','NOV','DEC']

def try_date(fetcher, dt):
    dstr = dt.strftime('%d%m%Y')
    mon  = MONTH_ABBR[dt.month - 1]
    old_dstr = dt.strftime('%d') + mon + dt.strftime('%Y')

    urls = [
        # New format
        f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{dstr}.csv",
        f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{dstr}.csv",
        # Old format
        f"https://nsearchives.nseindia.com/archives/equities/bhavcopy/cm{old_dstr}bhav.csv.zip",
        f"https://archives.nseindia.com/archives/equities/bhavcopy/cm{old_dstr}bhav.csv.zip",
    ]

    for url in urls:
        try:
            print(f"  Trying: {url}")
            page = fetcher.fetch(url)
            status = page.status
            size   = len(page.content) if page.content else 0
            print(f"    → HTTP {status}  size={size}")
            if status == 200 and size > 1000:
                print(f"  ✓ SUCCESS on {dt.date()} via {url}")
                return page.content
        except Exception as e:
            print(f"    → EXC: {e}")
    return None


def main():
    print("Installing/checking playwright for StealthyFetcher...")
    try:
        fetcher = StealthyFetcher(headless=True)
    except Exception as e:
        print(f"StealthyFetcher init failed: {e}")
        sys.exit(1)

    # Try last 10 business days
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=10)
    for dt in reversed(dates):
        print(f"\n--- {dt.date()} ---")
        raw = try_date(fetcher, dt)
        if raw:
            # Try to parse
            try:
                df = pd.read_csv(io.BytesIO(raw), low_memory=False)
                print(f"  Parsed: {len(df)} rows, cols: {list(df.columns[:6])}")
            except Exception as e:
                print(f"  Parse failed: {e} (may be zip)")
            break
    else:
        print("\nAll dates failed.")


if __name__ == '__main__':
    main()
