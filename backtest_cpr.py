"""
CPR Strategy Backtest — All 11 Rules
Data: Nifty 500 daily OHLCV
Entry: D+1 open  |  Exit: D+1 close (T+1 hold)
"""
# [STANDALONE] One-off analysis script — not part of the ML training pipeline.


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DATA_FILE = r"C:\Users\drkkr\Downloads\NIFTY 500 OHLCV\ALL_SYMBOLS_OHLCV.csv"
NARROW_THRESH = 0.5   # % for R2
MIN_PRICE     = 20    # skip penny stocks
MIN_BARS      = 12    # minimum bars needed before first signal
HOLD          = 1     # T+1: enter D+1 open, exit D+1 close
MAX_SPREAD    = 0.03  # skip if open-to-close spread > 3x avg (data error)

# ── INDICATORS ────────────────────────────────────────────────────────────────

def calc_cpr(H, L, C):
    pivot = (H + L + C) / 3
    bc    = (H + L) / 2
    tc    = 2 * pivot - bc
    upper = max(tc, bc)
    lower = min(tc, bc)
    width     = upper - lower
    width_pct = (width / pivot * 100) if pivot > 0 else 0
    return dict(pivot=pivot, upper=upper, lower=lower,
                width=width, width_pct=width_pct)

def calc_cam(H, L, C):
    r = H - L
    return dict(
        r4=C+r*1.1/2,  r3=C+r*1.1/4,
        r2=C+r*1.1/6,  r1=C+r*1.1/12,
        s1=C-r*1.1/12, s2=C-r*1.1/6,
        s3=C-r*1.1/4,  s4=C-r*1.1/2
    )

def calc_vwap(closes, highs, lows, vols):
    tp  = (highs + lows + closes) / 3
    sv  = (tp * vols).sum()
    tv  = vols.sum()
    return sv / tv if tv > 0 else closes.iloc[-1]

def aggregate(rows):
    return (rows['HIGH'].max(), rows['LOW'].min(),
            rows['CLOSE'].iloc[-1], rows['OPEN'].iloc[0])

# ── RULE CHECKS ───────────────────────────────────────────────────────────────

def check_all_rules(cpr, cam, prev_close, cur_close,
                    period_high, period_low, vwap,
                    or_high, or_low):
    R = {}
    # R1: Cam S3 OR R3 inside CPR
    s3_in = cpr['lower'] <= cam['s3'] <= cpr['upper']
    r3_in = cpr['lower'] <= cam['r3'] <= cpr['upper']
    R['rule1'] = s3_in or r3_in

    # R2: Narrow CPR
    R['rule2'] = cpr['width_pct'] < NARROW_THRESH

    # R3: Close crossing above TC
    R['rule3'] = prev_close < cpr['upper'] and cur_close > cpr['upper']

    # R4: Virgin CPR (current period hasn't touched CPR zone)
    R['rule4'] = period_high < cpr['lower'] or period_low > cpr['upper']

    # R5: CPR + VWAP confluence
    margin = max(cpr['width'] * 0.5, cpr['pivot'] * 0.002)
    R['rule5'] = (cpr['lower'] - margin) <= vwap <= (cpr['upper'] + margin)

    # R6: Wide CPR + Cam extreme
    safe_close = cur_close if cur_close > 0 else 1
    nr3 = abs(cur_close - cam['r3']) / safe_close < 0.005
    ns3 = abs(cur_close - cam['s3']) / safe_close < 0.005
    R['rule6'] = cpr['width_pct'] > 0.7 and (nr3 or ns3)

    # R7: CPR S/R Flip Retest
    if cpr['upper'] > 0 and cpr['lower'] > 0:
        ret_sup = (prev_close > cpr['upper'] and cur_close > cpr['upper']
                   and (cur_close - cpr['upper']) / cpr['upper'] < 0.012)
        ret_res = (prev_close < cpr['lower'] and cur_close < cpr['lower']
                   and (cpr['lower'] - cur_close) / cpr['lower'] < 0.012)
        R['rule7'] = ret_sup or ret_res
    else:
        R['rule7'] = False

    # R8: Opening Range + CPR aligned
    R['rule8'] = ((cur_close > or_high and cur_close > cpr['upper']) or
                  (cur_close < or_low  and cur_close < cpr['lower']))

    # R9: Pivot Magnetic Pull (>2% away)
    R['rule9'] = (cpr['pivot'] > 0 and
                  abs(cur_close - cpr['pivot']) / cpr['pivot'] > 0.02)

    # R10: Price testing CPR as S/R (within 0.5%)
    ht = cpr['upper'] > 0 and abs(period_high - cpr['upper']) / cpr['upper'] < 0.005
    lt = cpr['lower'] > 0 and abs(period_low  - cpr['lower']) / cpr['lower'] < 0.005
    R['rule10'] = ht or lt

    # R11: VWAP-to-TC setup (sandwiched)
    R['rule11'] = ((cur_close > vwap and cur_close < cpr['upper']) or
                   (cur_close < vwap and cur_close > cpr['lower']))

    return R

# ── DIRECTION ─────────────────────────────────────────────────────────────────

def get_direction(rule_id, cpr, cam, cur_close, vwap, prev_close,
                  period_high, period_low):
    """Returns +1 (long), -1 (short)."""
    if rule_id == 'rule3':   return 1      # always long: crossed above TC
    if rule_id == 'rule4':   return 1 if period_low > cpr['upper'] else -1
    if rule_id == 'rule6':   # near R3 = short; near S3 = long
        safe = cur_close if cur_close > 0 else 1
        return -1 if abs(cur_close - cam['r3']) / safe < 0.005 else 1
    if rule_id == 'rule7':   # support retest = long; resistance = short
        return 1 if prev_close > cpr['upper'] else -1
    if rule_id == 'rule8':   return 1 if cur_close > cpr['upper'] else -1
    if rule_id == 'rule9':   return -1 if cur_close > cpr['pivot'] else 1
    if rule_id == 'rule10':  # high tests upper = short; low tests lower = long
        ht = cpr['upper'] > 0 and abs(period_high - cpr['upper']) / cpr['upper'] < 0.005
        return -1 if ht else 1
    # Default for R1, R2, R5, R11: price vs pivot
    return 1 if cur_close >= cpr['pivot'] else -1

# ── METRICS ───────────────────────────────────────────────────────────────────

def compute_metrics(rets):
    rets = np.array(rets)
    if len(rets) == 0:
        return dict(trades=0, win_rate=0, pf=0, avg_ret=0, sharpe=0)
    wins  = rets[rets > 0]
    loss  = rets[rets <= 0]
    pf    = (wins.sum() / abs(loss.sum())) if loss.sum() != 0 else np.inf
    avg   = rets.mean() * 100
    std   = rets.std()
    sharp = (rets.mean() / std * np.sqrt(252)) if std > 0 else 0
    return dict(
        trades   = len(rets),
        win_rate = round(len(wins) / len(rets) * 100, 1),
        pf       = round(pf, 2),
        avg_ret  = round(avg, 3),
        sharpe   = round(sharp, 2)
    )

# ── MAIN BACKTEST ─────────────────────────────────────────────────────────────

print("Loading data…")
df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.upper()
df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%b-%Y')
df = df.sort_values(['SYMBOL', 'DATE']).reset_index(drop=True)
df['CLOSE'] = pd.to_numeric(df['CLOSE'], errors='coerce')
df['HIGH']  = pd.to_numeric(df['HIGH'],  errors='coerce')
df['LOW']   = pd.to_numeric(df['LOW'],   errors='coerce')
df['OPEN']  = pd.to_numeric(df['OPEN'],  errors='coerce')
df['VOLUME']= pd.to_numeric(df['VOLUME'],errors='coerce').fillna(0)
df = df.dropna(subset=['CLOSE','HIGH','LOW','OPEN'])

symbols   = df['SYMBOL'].unique()
rule_ids  = [f'rule{i}' for i in range(1, 12)]
all_trades = {r: [] for r in rule_ids}

print(f"Symbols: {len(symbols)} | Rows: {len(df):,}")
print("Running backtest…")

processed = 0
for sym, grp in df.groupby('SYMBOL'):
    grp = grp.reset_index(drop=True)
    n   = len(grp)
    if n < MIN_BARS + HOLD + 1:
        continue

    closes  = grp['CLOSE'].values
    highs   = grp['HIGH'].values
    lows    = grp['LOW'].values
    opens   = grp['OPEN'].values
    vols    = grp['VOLUME'].values

    # Need at least 6 bars for prevPeriod (5) + 1 current
    for i in range(6, n - HOLD - 1):
        entry_open  = opens[i + 1]
        entry_close = closes[i + 1]

        if entry_open <= 0 or entry_close <= 0:
            continue
        if closes[i] < MIN_PRICE:
            continue

        # prevPeriod: 5 bars [i-5 .. i-1]
        pH = highs[i-5:i].max()
        pL = lows[i-5:i].min()
        pC = closes[i-1]

        # currentBars for this week: [i-4 .. i] (5 bars incl today)
        cur_H = highs[i-4:i+1].max()
        cur_L = lows[i-4:i+1].min()

        # VWAP over current 5 bars
        h5 = pd.Series(highs[i-4:i+1])
        l5 = pd.Series(lows[i-4:i+1])
        c5 = pd.Series(closes[i-4:i+1])
        v5 = pd.Series(vols[i-4:i+1])
        vwap = calc_vwap(c5, h5, l5, v5)

        # Opening range (first 3 bars of current week)
        or_high = highs[i-4:i-1].max()
        or_low  = lows[i-4:i-1].min()

        cpr  = calc_cpr(pH, pL, pC)
        cam  = calc_cam(pH, pL, pC)
        prev_close  = closes[i-1]
        cur_close   = closes[i]

        rule_results = check_all_rules(
            cpr, cam, prev_close, cur_close,
            cur_H, cur_L, vwap, or_high, or_low
        )

        for rid in rule_ids:
            if not rule_results[rid]:
                continue
            direction = get_direction(rid, cpr, cam, cur_close, vwap,
                                      prev_close, cur_H, cur_L)
            ret = direction * (entry_close - entry_open) / entry_open
            all_trades[rid].append(ret)

    processed += 1
    if processed % 100 == 0:
        print(f"  {processed}/{len(symbols)} symbols…")

# ── RESULTS ───────────────────────────────────────────────────────────────────

RULE_NAMES = {
    'rule1':  'R1  Cam S3&R3 Inside CPR',
    'rule2':  'R2  Narrow CPR',
    'rule3':  'R3  Cross Above TC',
    'rule4':  'R4  Virgin CPR',
    'rule5':  'R5  CPR + VWAP Confluence',
    'rule6':  'R6  Wide CPR + Cam Extreme',
    'rule7':  'R7  CPR S/R Flip Retest',
    'rule8':  'R8  OR + CPR Aligned',
    'rule9':  'R9  Pivot Magnetic Pull',
    'rule10': 'R10 Price Testing CPR S/R',
    'rule11': 'R11 VWAP-to-TC Setup',
}

print()
print("=" * 75)
print(f"{'Strategy':<30} {'Trades':>7} {'Win%':>7} {'PF':>7} {'Avg%':>8} {'Sharpe':>8}")
print("=" * 75)

summary = []
for rid in rule_ids:
    m = compute_metrics(all_trades[rid])
    summary.append({**m, 'rule': rid, 'name': RULE_NAMES[rid]})
    print(f"{RULE_NAMES[rid]:<30} {m['trades']:>7} {m['win_rate']:>7} {m['pf']:>7} {m['avg_ret']:>8} {m['sharpe']:>8}")

print("=" * 75)

# ── SAVE TO CSV ───────────────────────────────────────────────────────────────
out = pd.DataFrame(summary)[['name','trades','win_rate','pf','avg_ret','sharpe']]
out.columns = ['Strategy','Trades','WinRate%','ProfitFactor','AvgReturn%','Sharpe']
out.to_csv(r"D:\Claude code\nse-screener\backtest_results.csv", index=False)
print(f"\nResults saved → backtest_results.csv")
print(f"Data period: Jun 2021 – Jun 2026 | Hold: T+1 (next day open→close)")
