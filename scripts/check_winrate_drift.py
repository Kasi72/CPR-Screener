"""
Win-rate drift monitor — compares 30-day rolling live win rate from UC logger
against the training baseline (29.4%). Alerts if drift exceeds threshold.

Usage:
    python scripts/check_winrate_drift.py

Requirements:
    NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env.local
    (or set as env vars directly)

Exit codes:
    0 = healthy (win rate within tolerance)
    1 = drift detected (retrain recommended)
    2 = insufficient data (<50 labelled trades in window)
"""

import os, sys, json
from datetime import datetime, timedelta
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
TRAINING_WIN_RATE = 0.294       # hit_t1 baseline from last retrain
DRIFT_THRESHOLD   = 0.05        # alert if |live - baseline| > 5 pp
MIN_TRADES        = 50          # minimum labelled trades to compute drift
LOOKBACK_DAYS     = 30

# ── Load env ──────────────────────────────────────────────────────────────────
env_path = Path(__file__).parents[1] / '.env.local'
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            k, _, v = line.partition('=')
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

try:
    from supabase import create_client
except ImportError:
    print("ERROR: pip install supabase")
    sys.exit(2)

url = os.environ.get('NEXT_PUBLIC_SUPABASE_URL')
key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
if not url or not key:
    print("ERROR: NEXT_PUBLIC_SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set")
    sys.exit(2)

# ── Query UC logger for labelled outcomes ──────────────────────────────────────
sb = create_client(url, key)
since = (datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)).isoformat()

try:
    rows = (
        sb.table('uc_scan_log')
        .select('hit_t1, scan_date')
        .gte('scan_date', since)
        .not_.is_('hit_t1', 'null')   # only labelled outcomes
        .execute()
    ).data
except Exception as e:
    print(f"ERROR querying uc_scan_log: {e}")
    sys.exit(2)

if not rows:
    print(f"No labelled trades in last {LOOKBACK_DAYS} days — check uc_scan_log table name.")
    sys.exit(2)

n_total = len(rows)
n_wins  = sum(1 for r in rows if r.get('hit_t1'))
live_wr = n_wins / n_total

drift   = live_wr - TRAINING_WIN_RATE
status  = 'HEALTHY' if abs(drift) <= DRIFT_THRESHOLD else 'DRIFT DETECTED'

print("=" * 55)
print("  Win-Rate Drift Monitor")
print("=" * 55)
print(f"  Window:          last {LOOKBACK_DAYS} days")
print(f"  Labelled trades: {n_total}")
print(f"  Live win rate:   {live_wr:.1%}")
print(f"  Training baseline: {TRAINING_WIN_RATE:.1%}")
print(f"  Drift:           {drift:+.1%}")
print(f"  Status:          {status}")
print("=" * 55)

if n_total < MIN_TRADES:
    print(f"  WARNING: only {n_total} trades — need ≥{MIN_TRADES} for reliable estimate.")
    sys.exit(2)

if abs(drift) > DRIFT_THRESHOLD:
    print(f"\n  *** RETRAIN RECOMMENDED ***")
    print(f"  Win rate dropped {abs(drift):.1%} below baseline.")
    print(f"  Run: python scripts/ml/run_all.py --skip-dataset")
    sys.exit(1)

print(f"\n  Model healthy. Next check: {(datetime.utcnow() + timedelta(days=7)).strftime('%Y-%m-%d')}")
sys.exit(0)
