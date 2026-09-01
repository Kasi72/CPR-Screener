"""
NSE bhavcopy download — Playwright cookie bridge + requests download.
archives.nseindia.com serves the file (confirmed "Download is starting").
Strategy: get real browser cookies from Playwright, then use requests to download.
"""
# [STANDALONE] One-off analysis script — not part of the ML training pipeline.

import sys, io, time
import pandas as pd
import requests
from playwright.sync_api import sync_playwright

MONTH_ABBR = ['JAN','FEB','MAR','APR','MAY','JUN',
               'JUL','AUG','SEP','OCT','NOV','DEC']

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')


def get_cookies_via_playwright():
    """Visit NSE + archives domain to acquire real session cookies."""
    cookies = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=UA,
            locale='en-IN',
            timezone_id='Asia/Kolkata',
            accept_downloads=True,
        )
        page = ctx.new_page()

        # Step 1: visit archives domain directly (sometimes enough)
        print("  [1] Hitting archives.nseindia.com...")
        try:
            page.goto('https://archives.nseindia.com', timeout=15000,
                      wait_until='domcontentloaded')
            page.wait_for_timeout(2000)
        except Exception as e:
            print(f"  WARN: {e}")

        # Step 2: visit NSE main (bypasses main domain JS gate)
        print("  [2] Hitting nseindia.com...")
        try:
            page.goto('https://www.nseindia.com', timeout=20000,
                      wait_until='domcontentloaded')
            page.wait_for_timeout(3000)
        except Exception as e:
            print(f"  WARN: {e}")

        raw_cookies = ctx.cookies()
        for c in raw_cookies:
            cookies[c['name']] = c['value']
        browser.close()

    print(f"  Got {len(cookies)} cookies: {list(cookies.keys())[:10]}")
    return cookies


def download_with_requests(url, cookies):
    """Download file using requests + browser cookies."""
    headers = {
        'User-Agent': UA,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-IN,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.nseindia.com/',
        'Connection': 'keep-alive',
    }
    try:
        r = requests.get(url, headers=headers, cookies=cookies, timeout=30, stream=True)
        print(f"    HTTP {r.status_code}  Content-Type: {r.headers.get('Content-Type','?')}  Size: {r.headers.get('Content-Length','?')}")
        if r.status_code == 200:
            data = r.content
            if len(data) > 1000:
                return data
    except Exception as e:
        print(f"    EXC: {e}")
    return None


def try_playwright_download_event(dt):
    """Use Playwright download event listener to capture file bytes."""
    dstr = dt.strftime('%d%m%Y')
    url  = f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{dstr}.csv"
    print(f"  Playwright download event: {url}")

    result = {'data': None}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=UA, accept_downloads=True)
        page = ctx.new_page()

        downloads_received = []

        def on_download(download):
            print(f"  Download event: {download.suggested_filename}")
            path = download.path()
            if path:
                with open(path, 'rb') as f:
                    downloads_received.append(f.read())
                download.delete()

        page.on('download', on_download)

        # Navigate — it will raise but the download event fires first
        try:
            page.goto(url, timeout=20000)
        except Exception as e:
            msg = str(e)
            if 'Download is starting' in msg:
                print("  Got expected 'Download is starting' — checking event queue...")
            else:
                print(f"  goto EXC: {msg[:100]}")

        # Wait a moment for the download event to complete
        page.wait_for_timeout(5000)
        browser.close()

    if downloads_received:
        return downloads_received[0]
    return None


def main():
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=5)

    # Get cookies first
    print("=== Acquiring NSE cookies via Playwright ===")
    cookies = get_cookies_via_playwright()

    for dt in reversed(dates):
        dstr    = dt.strftime('%d%m%Y')
        mon     = MONTH_ABBR[dt.month - 1]
        old_str = dt.strftime('%d') + mon + dt.strftime('%Y')

        print(f"\n=== {dt.date()} ===")

        urls = [
            f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{dstr}.csv",
            f"https://archives.nseindia.com/archives/equities/bhavcopy/cm{old_str}bhav.csv.zip",
        ]

        raw = None
        for url in urls:
            print(f"  GET {url}")
            raw = download_with_requests(url, cookies)
            if raw:
                print(f"  Downloaded: {len(raw):,} bytes")
                break

        if not raw:
            # Fallback: try Playwright download event
            print("  Trying Playwright download event interception...")
            raw = try_playwright_download_event(dt)

        if raw:
            print(f"  Total size: {len(raw):,} bytes")
            try:
                content = raw
                # Handle zip
                if raw[:2] == b'PK':
                    import zipfile
                    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                        content = zf.read(zf.namelist()[0])
                df = pd.read_csv(io.BytesIO(content), low_memory=False)
                df.columns = [c.strip().upper().replace(' ','_') for c in df.columns]
                print(f"  Rows: {len(df)}  Cols: {list(df.columns[:8])}")
                for col in ['DELIV_PER', 'DELIVERY_QTY']:
                    if col in df.columns:
                        print(f"  ✓ DELIVERY COLUMN FOUND: {col}")
                        break
                print(df.head(3).to_string())
            except Exception as e:
                print(f"  Parse error: {e}")
            break
    else:
        print("\nAll dates failed — NSE bhavcopy not accessible via automation.")


if __name__ == '__main__':
    main()
