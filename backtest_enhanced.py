"""
CPR Enhanced Backtest v2 — All 11 Rules
Layers: SG filter + Kalman + 200-EMA + ADX + ATR percentile + Volume rank
        + Signal confluence + Hurst exponent + Calendar filter
Exit:   Asymmetric T+1/T+3 hold, trailing stop, profit target
"""
# [STANDALONE] One-off analysis script — not part of the ML training pipeline.


import pandas as pd
import numpy as np
import warnings
from scipy.signal import savgol_filter
warnings.filterwarnings('ignore')

DATA_FILE    = r"C:\Users\drkkr\Downloads\NIFTY 500 OHLCV\ALL_SYMBOLS_OHLCV.csv"
NARROW_THRESH = 0.5
MIN_PRICE     = 20
MIN_BARS      = 60   # need enough history for 200-EMA, ADX, ATR percentile
MAX_HOLD      = 3    # asymmetric: up to T+3 for winners

# ── SAVITZKY-GOLAY ────────────────────────────────────────────────────────────

def sg_velocity(closes, window=11, poly=3):
    """1st derivative = trend velocity. Positive = uptrend."""
    if len(closes) < window:
        return 0.0
    return float(savgol_filter(closes, window, poly, deriv=1)[-1])

def sg_acceleration(closes, window=11, poly=3):
    """2nd derivative = momentum curvature. Positive = accelerating up."""
    if len(closes) < window:
        return 0.0
    return float(savgol_filter(closes, window, poly, deriv=2)[-1])

# ── KALMAN FILTER ─────────────────────────────────────────────────────────────

def kalman_smooth(closes, Q=1e-3, R=0.1):
    """
    Simple 1D Kalman (price state).
    Returns (smoothed_price, kalman_velocity_last, kalman_gain_last).
    Kalman gain high = high uncertainty = volatile regime.
    """
    n   = len(closes)
    x   = closes[0]
    P   = 1.0
    xs  = np.zeros(n)
    ks  = np.zeros(n)
    for i, z in enumerate(closes):
        P += Q
        K  = P / (P + R)
        x  = x + K * (z - x)
        P  = (1 - K) * P
        xs[i] = x
        ks[i] = K
    velocity = xs[-1] - xs[-2] if n > 1 else 0.0
    return xs, velocity, ks[-1]

# ── INDICATORS ────────────────────────────────────────────────────────────────

def ema(arr, span):
    s = pd.Series(arr)
    return s.ewm(span=span, adjust=False).mean().values

def atr(highs, lows, closes, period=14):
    h = np.array(highs); l = np.array(lows); c = np.array(closes)
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(np.abs(h[1:] - c[:-1]),
                    np.abs(l[1:] - c[:-1])))
    if len(tr) < period:
        return np.zeros(len(closes))
    atr_vals = np.zeros(len(closes))
    atr_vals[period] = tr[:period].mean()
    for i in range(period + 1, len(closes)):
        atr_vals[i] = (atr_vals[i-1] * (period - 1) + tr[i-1]) / period
    return atr_vals

def adx_series(highs, lows, closes, period=14):
    h = np.array(highs); l = np.array(lows); c = np.array(closes)
    n  = len(c)
    dm_plus  = np.zeros(n)
    dm_minus = np.zeros(n)
    tr_arr   = np.zeros(n)
    for i in range(1, n):
        up   = h[i] - h[i-1]
        down = l[i-1] - l[i]
        dm_plus[i]  = up   if up > down and up > 0 else 0
        dm_minus[i] = down if down > up and down > 0 else 0
        tr_arr[i]   = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    def smooth(x):
        s = np.zeros(n)
        s[period] = x[1:period+1].sum()
        for i in range(period + 1, n):
            s[i] = s[i-1] - s[i-1] / period + x[i]
        return s
    str14   = smooth(tr_arr)
    sdm_p   = smooth(dm_plus)
    sdm_m   = smooth(dm_minus)
    di_p = np.where(str14 > 0, 100 * sdm_p / str14, 0)
    di_m = np.where(str14 > 0, 100 * sdm_m / str14, 0)
    dx   = np.where(di_p + di_m > 0, 100 * np.abs(di_p - di_m) / (di_p + di_m), 0)
    adx  = np.zeros(n)
    start = 2 * period
    if n > start:
        adx[start] = dx[period:start+1].mean()
        for i in range(start + 1, n):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    return adx

def hurst_exponent(series, min_lags=20):
    """Simplified R/S Hurst. H>0.55=trending, H<0.45=mean-reverting."""
    ts = np.log(series / series[0] + 1e-10)
    lags = range(2, min(min_lags, len(series) // 2))
    rs_list = []
    for lag in lags:
        chunks = [ts[j:j+lag] for j in range(0, len(ts) - lag, lag)]
        rs = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean = chunk.mean()
            dev  = np.cumsum(chunk - mean)
            R    = dev.max() - dev.min()
            S    = chunk.std()
            if S > 0:
                rs.append(R / S)
        if rs:
            rs_list.append((np.log(lag), np.log(np.mean(rs))))
    if len(rs_list) < 4:
        return 0.5
    lags_log, rs_log = zip(*rs_list)
    H = np.polyfit(lags_log, rs_log, 1)[0]
    return float(np.clip(H, 0.1, 0.9))

# ── CPR / CAM / VWAP (same as v1) ─────────────────────────────────────────────

def calc_cpr(H, L, C):
    pivot = (H + L + C) / 3
    bc    = (H + L) / 2
    tc    = 2 * pivot - bc
    upper = max(tc, bc); lower = min(tc, bc)
    width     = upper - lower
    width_pct = (width / pivot * 100) if pivot > 0 else 0
    return dict(pivot=pivot, upper=upper, lower=lower,
                width=width, width_pct=width_pct)

def calc_cam(H, L, C):
    r = H - L
    return dict(r4=C+r*1.1/2, r3=C+r*1.1/4,
                r2=C+r*1.1/6, r1=C+r*1.1/12,
                s1=C-r*1.1/12, s2=C-r*1.1/6,
                s3=C-r*1.1/4,  s4=C-r*1.1/2)

def calc_vwap(c5, h5, l5, v5):
    tp = (h5 + l5 + c5) / 3
    sv = (tp * v5).sum(); tv = v5.sum()
    return sv / tv if tv > 0 else c5[-1]

# ── RULES (identical logic to v1) ─────────────────────────────────────────────

def check_all_rules(cpr, cam, prev_close, cur_close,
                    period_high, period_low, vwap, or_high, or_low):
    R = {}
    s3_in = cpr['lower'] <= cam['s3'] <= cpr['upper']
    r3_in = cpr['lower'] <= cam['r3'] <= cpr['upper']
    R['rule1'] = s3_in or r3_in
    R['rule2'] = cpr['width_pct'] < NARROW_THRESH
    R['rule3'] = prev_close < cpr['upper'] and cur_close > cpr['upper']
    R['rule4'] = period_high < cpr['lower'] or period_low > cpr['upper']
    margin = max(cpr['width'] * 0.5, cpr['pivot'] * 0.002)
    R['rule5'] = (cpr['lower'] - margin) <= vwap <= (cpr['upper'] + margin)
    safe = cur_close if cur_close > 0 else 1
    nr3  = abs(cur_close - cam['r3']) / safe < 0.005
    ns3  = abs(cur_close - cam['s3']) / safe < 0.005
    R['rule6'] = cpr['width_pct'] > 0.7 and (nr3 or ns3)
    if cpr['upper'] > 0 and cpr['lower'] > 0:
        ret_sup = (prev_close > cpr['upper'] and cur_close > cpr['upper']
                   and (cur_close - cpr['upper']) / cpr['upper'] < 0.012)
        ret_res = (prev_close < cpr['lower'] and cur_close < cpr['lower']
                   and (cpr['lower'] - cur_close) / cpr['lower'] < 0.012)
        R['rule7'] = ret_sup or ret_res
    else:
        R['rule7'] = False
    R['rule8']  = ((cur_close > or_high and cur_close > cpr['upper']) or
                   (cur_close < or_low  and cur_close < cpr['lower']))
    R['rule9']  = (cpr['pivot'] > 0 and
                   abs(cur_close - cpr['pivot']) / cpr['pivot'] > 0.02)
    ht = cpr['upper'] > 0 and abs(period_high - cpr['upper']) / cpr['upper'] < 0.005
    lt = cpr['lower'] > 0 and abs(period_low  - cpr['lower']) / cpr['lower'] < 0.005
    R['rule10'] = ht or lt
    R['rule11'] = ((cur_close > vwap and cur_close < cpr['upper']) or
                   (cur_close < vwap and cur_close > cpr['lower']))
    return R

def get_direction(rule_id, cpr, cam, cur_close, vwap, prev_close, ph, pl):
    if rule_id == 'rule3':  return 1
    if rule_id == 'rule4':  return 1 if pl > cpr['upper'] else -1
    if rule_id == 'rule6':
        safe = cur_close if cur_close > 0 else 1
        return -1 if abs(cur_close - cam['r3']) / safe < 0.005 else 1
    if rule_id == 'rule7':  return 1 if prev_close > cpr['upper'] else -1
    if rule_id == 'rule8':  return 1 if cur_close > cpr['upper'] else -1
    if rule_id == 'rule9':  return -1 if cur_close > cpr['pivot'] else 1
    if rule_id == 'rule10':
        ht = cpr['upper'] > 0 and abs(ph - cpr['upper']) / cpr['upper'] < 0.005
        return -1 if ht else 1
    return 1 if cur_close >= cpr['pivot'] else -1

# ── METRICS ───────────────────────────────────────────────────────────────────

def compute_metrics(rets):
    rets = np.array(rets)
    if len(rets) == 0:
        return dict(trades=0, win_rate=0, pf=0, avg_ret=0, sharpe=0)
    wins = rets[rets > 0]; loss = rets[rets <= 0]
    pf   = (wins.sum() / abs(loss.sum())) if loss.sum() != 0 else np.inf
    std  = rets.std()
    return dict(
        trades   = len(rets),
        win_rate = round(len(wins) / len(rets) * 100, 1),
        pf       = round(pf, 2),
        avg_ret  = round(rets.mean() * 100, 3),
        sharpe   = round(rets.mean() / std * np.sqrt(252) if std > 0 else 0, 2)
    )

# ── ASYMMETRIC EXIT ───────────────────────────────────────────────────────────

def simulate_asymmetric_exit(direction, entry_open, highs_fwd, lows_fwd,
                              closes_fwd, profit_target=0.015, trail_stop=0.012):
    """
    Hold up to MAX_HOLD bars. Exit rules (in priority order):
    1. Profit target hit (+1.5% from entry) -> lock gain
    2. Trailing stop (1.2% drawdown from highest unrealised profit)
    3. T+1 close if in loss -> cut early
    4. T+3 close if still running
    Returns: final return as fraction.
    """
    if entry_open <= 0:
        return 0.0

    peak_pnl = 0.0

    for day in range(min(MAX_HOLD, len(closes_fwd))):
        h = highs_fwd[day]; l = lows_fwd[day]; c = closes_fwd[day]
        if h <= 0 or c <= 0:
            break

        # intraday best/worst for direction
        if direction == 1:
            best_pnl  = (h - entry_open) / entry_open
            worst_pnl = (l - entry_open) / entry_open
        else:
            best_pnl  = (entry_open - l) / entry_open
            worst_pnl = (entry_open - h) / entry_open

        # update peak
        peak_pnl = max(peak_pnl, best_pnl)

        # profit target hit intraday
        if best_pnl >= profit_target:
            return profit_target * 0.95  # haircut for limit fill slippage

        # trailing stop from peak
        if peak_pnl > 0 and (peak_pnl - best_pnl) >= trail_stop:
            return peak_pnl - trail_stop

        # T+1 loss cut: if day 0 closes in loss, exit at close
        day_ret = direction * (c - entry_open) / entry_open
        if day == 0 and day_ret < 0:
            return day_ret

    # exit at last available close
    c_last = closes_fwd[min(MAX_HOLD, len(closes_fwd)) - 1]
    return direction * (c_last - entry_open) / entry_open

# ── CALENDAR FILTER ───────────────────────────────────────────────────────────

def is_bad_calendar(date):
    """Avoid Mondays, Fridays. True = skip signal."""
    dow = date.weekday()  # 0=Mon, 4=Fri
    return dow == 0 or dow == 4

# ── MAIN ──────────────────────────────────────────────────────────────────────

print("Loading data…")
df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.upper()
df['DATE']   = pd.to_datetime(df['DATE'], format='%d-%b-%Y')
df           = df.sort_values(['SYMBOL', 'DATE']).reset_index(drop=True)
for col in ['CLOSE', 'HIGH', 'LOW', 'OPEN']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['VOLUME'] = pd.to_numeric(df['VOLUME'], errors='coerce').fillna(0)
df = df.dropna(subset=['CLOSE', 'HIGH', 'LOW', 'OPEN'])

rule_ids = [f'rule{i}' for i in range(1, 12)]

# Two result sets: base (filters only) and enhanced (all filters + ML-ready saves)
base_trades     = {r: [] for r in rule_ids}
enhanced_trades = {r: [] for r in rule_ids}

symbols = df['SYMBOL'].unique()
print(f"Symbols: {len(symbols):,} | Rows: {len(df):,}")
print("Running enhanced backtest…\n")

processed = 0
skipped   = 0

for sym, grp in df.groupby('SYMBOL'):
    grp = grp.reset_index(drop=True)
    n   = len(grp)
    if n < MIN_BARS + MAX_HOLD + 2:
        skipped += 1
        continue

    closes = grp['CLOSE'].values
    highs  = grp['HIGH'].values
    lows   = grp['LOW'].values
    opens  = grp['OPEN'].values
    vols   = grp['VOLUME'].values
    dates  = grp['DATE'].values

    # Pre-compute series-level indicators
    ema200   = ema(closes, 200)
    adx_vals = adx_series(highs, lows, closes, 14)
    atr_vals = atr(highs, lows, closes, 14)
    vol20    = pd.Series(vols).rolling(20).mean().values

    for i in range(55, n - MAX_HOLD - 2):
        if closes[i] < MIN_PRICE:
            continue

        cur_date = pd.Timestamp(dates[i])
        if is_bad_calendar(cur_date):
            continue

        # ATR percentile filter (35th–75th percentile over trailing 120 bars)
        atr_window = atr_vals[max(0, i-120):i]
        atr_window = atr_window[atr_window > 0]
        if len(atr_window) < 30:
            continue
        atr_now = atr_vals[i]
        if atr_now <= 0:
            continue
        pct_rank = np.sum(atr_window <= atr_now) / len(atr_window)
        if not (0.35 <= pct_rank <= 0.75):
            continue

        # ADX > 20 filter
        if adx_vals[i] < 20:
            continue

        # Volume filter: today > 1.5× 20-day avg
        if vol20[i] > 0 and vols[i] < 1.5 * vol20[i]:
            continue

        # 200-EMA regime
        ema200_now = ema200[i]

        # SG velocity (use last 21 bars for context)
        sg_win = closes[max(0, i-20):i+1]
        sg_vel  = sg_velocity(sg_win)
        sg_accel = sg_acceleration(sg_win)

        # Kalman on last 60 bars
        kal_win = closes[max(0, i-59):i+1]
        _, kal_vel, kal_gain = kalman_smooth(kal_win)

        # Hurst on last 60 bars (routing only, not hard filter)
        hurst = hurst_exponent(closes[max(0, i-59):i+1])

        # prevPeriod: bars [i-5..i-1]
        pH = highs[i-5:i].max(); pL = lows[i-5:i].min(); pC = closes[i-1]
        # currentBars [i-4..i]
        cur_H = highs[i-4:i+1].max(); cur_L = lows[i-4:i+1].min()
        h5 = highs[i-4:i+1]; l5 = lows[i-4:i+1]
        c5 = closes[i-4:i+1]; v5 = vols[i-4:i+1]
        vwap       = calc_vwap(c5, h5, l5, v5)
        or_high    = highs[i-4:i-1].max(); or_low = lows[i-4:i-1].min()

        cpr        = calc_cpr(pH, pL, pC)
        cam        = calc_cam(pH, pL, pC)
        prev_close = closes[i-1]; cur_close = closes[i]

        rules_fired = check_all_rules(cpr, cam, prev_close, cur_close,
                                      cur_H, cur_L, vwap, or_high, or_low)

        # Confluence: require 2+ rules firing
        n_fired = sum(1 for r in rule_ids if rules_fired[r])
        if n_fired < 2:
            continue

        entry_open = opens[i+1]
        if entry_open <= 0:
            continue

        for rid in rule_ids:
            if not rules_fired[rid]:
                continue

            direction = get_direction(rid, cpr, cam, cur_close, vwap,
                                      prev_close, cur_H, cur_L)

            # ── BASE FILTER RETURN (T+1) ──────────────────────────────
            if i + 1 < n:
                entry_c = closes[i+1]
                if entry_c > 0:
                    base_ret = direction * (entry_c - entry_open) / entry_open
                    base_trades[rid].append(base_ret)

            # ── ENHANCED: TREND GATE ──────────────────────────────────
            # Long only above 200-EMA, short only below
            if direction == 1 and cur_close < ema200_now * 0.99:
                continue
            if direction == -1 and cur_close > ema200_now * 1.01:
                continue

            # SG velocity must align with direction
            if direction == 1 and sg_vel < 0:
                continue
            if direction == -1 and sg_vel > 0:
                continue

            # Kalman velocity must align
            if direction == 1 and kal_vel < 0:
                continue
            if direction == -1 and kal_vel > 0:
                continue

            # Hurst routing: breakout rules need trending, reversion rules need H<0.55
            breakout_rules  = {'rule3', 'rule4', 'rule8'}
            reversion_rules = {'rule7', 'rule10', 'rule11'}
            if rid in breakout_rules and hurst < 0.45:
                continue
            if rid in reversion_rules and hurst > 0.65:
                continue

            # ── ASYMMETRIC EXIT ───────────────────────────────────────
            max_fwd = min(MAX_HOLD, n - i - 2)
            if max_fwd < 1:
                continue
            fwd_highs  = highs[i+1:i+1+max_fwd]
            fwd_lows   = lows[i+1:i+1+max_fwd]
            fwd_closes = closes[i+1:i+1+max_fwd]

            enh_ret = simulate_asymmetric_exit(direction, entry_open,
                                               fwd_highs, fwd_lows, fwd_closes)
            enhanced_trades[rid].append(enh_ret)

    processed += 1
    if processed % 50 == 0:
        print(f"  {processed}/{len(symbols)-skipped} symbols…")

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
print("=" * 90)
print(f"{'Strategy':<30} {'Base Trades':>11} {'Base Win%':>9} {'Base PF':>8}"
      f" {'Enh Trades':>10} {'Enh Win%':>9} {'Enh PF':>8} {'Enh Sharpe':>10}")
print("=" * 90)

summary = []
for rid in rule_ids:
    bm = compute_metrics(base_trades[rid])
    em = compute_metrics(enhanced_trades[rid])
    row = {**{'rule': rid, 'name': RULE_NAMES[rid]},
           'base_trades': bm['trades'], 'base_win': bm['win_rate'],
           'base_pf': bm['pf'],
           'enh_trades': em['trades'], 'enh_win': em['win_rate'],
           'enh_pf': em['pf'], 'enh_avg': em['avg_ret'],
           'enh_sharpe': em['sharpe']}
    summary.append(row)
    print(f"{RULE_NAMES[rid]:<30} {bm['trades']:>11,} {bm['win_rate']:>9} {bm['pf']:>8}"
          f" {em['trades']:>10,} {em['win_rate']:>9} {em['pf']:>8} {em['sharpe']:>10}")

print("=" * 90)

# Save
out = pd.DataFrame(summary)
out.to_csv(r"D:\Claude code\nse-screener\backtest_enhanced_results.csv", index=False)
print(f"\nSaved → backtest_enhanced_results.csv")

# Also dump all trade-level returns for XGBoost training
all_rows = []
for rid in rule_ids:
    for ret in enhanced_trades[rid]:
        all_rows.append({'rule': rid, 'return': ret, 'win': int(ret > 0)})
trades_df = pd.DataFrame(all_rows)
trades_df.to_csv(r"D:\Claude code\nse-screener\enhanced_trade_returns.csv", index=False)
print(f"Trade returns → enhanced_trade_returns.csv  ({len(trades_df):,} rows)")
print(f"\nPeriod: Jun 2021 – Jun 2026 | Filters: SG+Kalman+EMA200+ADX+ATR%ile+Vol+Confluence")
print(f"Exit: Asymmetric T1-T3, profit target +1.5%, trailing stop 1.2%")
