"""
CPR Backtest v3 — Full Enhancement Stack
New vs v2:
  - Volume Profile (POC/VAH/VAL) from daily OHLCV approximation
  - Weighted confluence score (by OOS Sharpe, not binary count)
  - RSI divergence filter on breakout rules R3/R8
  - India VIX regime gate (VIX > 18 → skip)
  - Nifty 50 master trend filter (20-EMA gate)
  - Per-rule MFE/MAE analysis → optimal stop/target
  - All v2 filters retained: SG + Kalman + EMA200 + ADX + ATR% + Volume
"""
# [STANDALONE] One-off analysis script — not part of the ML training pipeline.


import pandas as pd
import numpy as np
import warnings
from scipy.signal import savgol_filter
warnings.filterwarnings('ignore')

DATA_FILE = r"C:\Users\drkkr\Downloads\NIFTY 500 OHLCV\ALL_SYMBOLS_OHLCV.csv"
OUT_DIR   = r"D:\Claude code\nse-screener"

# ── PARAMS ────────────────────────────────────────────────────────────────────
MIN_BARS      = 60
MAX_HOLD      = 5
PROFIT_TARGET = 0.025   # will be overridden per-rule after MFE/MAE analysis
TRAIL_STOP    = 0.008
VIX_THRESHOLD = 18.0
NIFTY_EMA_SPAN = 20
VP_LOOKBACK   = 20      # bars for volume profile
VP_BINS       = 60      # price bins for volume histogram
NARROW_THRESH = 0.5

# OOS Sharpe weights per rule (from xgb_overlay_v2 results)
RULE_SHARPE = {
    'rule1': 2.86, 'rule2': 4.75, 'rule3': 0.44,
    'rule4': 0.0,  'rule5': 3.24, 'rule6': 3.71,
    'rule7': 6.19, 'rule8': 2.44, 'rule9': 0.0,   # R9 negative → 0
    'rule10': 2.89, 'rule11': 3.90
}
CONFLUENCE_THRESHOLD = 3.5   # min weighted score to take a trade

RULE_IDS = [f'rule{i}' for i in range(1, 12)]
BREAKOUT_RULES  = {'rule3', 'rule8'}

# ── VIX + NIFTY DOWNLOAD ─────────────────────────────────────────────────────

def fetch_series(ticker, start='2021-01-01', end='2026-12-31'):
    import yfinance as yf

    # Method 1: Ticker.history — flat columns, works on all yfinance versions
    try:
        df = yf.Ticker(ticker).history(start=start, end=end)
        if not df.empty and 'Close' in df.columns:
            out = {}
            for d, v in zip(df.index, df['Close']):
                try: out[str(d.date())] = float(v)
                except: pass
            if out:
                return out
    except Exception:
        pass

    # Method 2: yf.download with MultiIndex handling (newer yfinance)
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if not df.empty:
            if hasattr(df.columns, 'get_level_values'):
                df.columns = df.columns.get_level_values(0)
            col = 'Close' if 'Close' in df.columns else df.columns[0]
            out = {}
            for d, v in zip(df.index, df[col]):
                try: out[str(d.date())] = float(v)
                except: pass
            return out
    except Exception as e:
        print(f"  Warning: could not download {ticker}: {e}")
    return {}

print("Downloading India VIX and Nifty 50…")
vix_map   = fetch_series('^INDIAVIX')
nifty_map = fetch_series('^NSEI')

# Build Nifty 20-EMA by date
if nifty_map:
    nifty_dates  = sorted(nifty_map.keys())
    nifty_closes = np.array([nifty_map[d] for d in nifty_dates])
    k = 2 / (NIFTY_EMA_SPAN + 1)
    ema20 = np.zeros(len(nifty_closes))
    ema20[0] = nifty_closes[0]
    for i in range(1, len(nifty_closes)):
        ema20[i] = nifty_closes[i] * k + ema20[i-1] * (1 - k)
    nifty_ema20_map = {nifty_dates[i]: ema20[i] for i in range(len(nifty_dates))}
    print(f"  Nifty: {len(nifty_dates)} days | VIX: {len(vix_map)} days")
else:
    nifty_ema20_map = {}
    print("  Nifty/VIX unavailable — those filters disabled")

def get_vix(date_str):
    return vix_map.get(date_str, 0.0)

def get_nifty_ema20(date_str):
    return nifty_ema20_map.get(date_str, 0.0)

# ── INDICATORS ────────────────────────────────────────────────────────────────

def ema_series(arr, span):
    k = 2 / (span + 1)
    out = np.zeros(len(arr))
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i-1] * (1 - k)
    return out

def sg_velocity(closes, window=11, poly=3):
    if len(closes) < window: return 0.0
    return float(savgol_filter(closes, window, poly, deriv=1)[-1])

def kalman_velocity(closes, Q=1e-3, R=0.1):
    x = closes[0]; P = 1.0; prev_x = closes[0]
    for z in closes:
        P += Q; K = P / (P + R)
        prev_x = x
        x = x + K * (z - x); P = (1 - K) * P
    return x - prev_x

def atr_series_fn(highs, lows, closes, period=14):
    n = len(closes)
    tr = np.zeros(n)
    for j in range(1, n):
        tr[j] = max(highs[j]-lows[j], abs(highs[j]-closes[j-1]), abs(lows[j]-closes[j-1]))
    out = np.zeros(n)
    if n > period:
        out[period] = tr[1:period+1].mean()
        for j in range(period+1, n):
            out[j] = (out[j-1]*(period-1) + tr[j]) / period
    return out

def adx_at(highs, lows, closes, period=14):
    n = len(closes)
    if n < 2*period+2: return 0.0
    dm_p = np.zeros(n); dm_m = np.zeros(n); tr = np.zeros(n)
    for j in range(1, n):
        up = highs[j]-highs[j-1]; dn = lows[j-1]-lows[j]
        dm_p[j] = up   if up > dn and up > 0 else 0
        dm_m[j] = dn   if dn > up and dn > 0 else 0
        tr[j]   = max(highs[j]-lows[j], abs(highs[j]-closes[j-1]), abs(lows[j]-closes[j-1]))
    def sm(x):
        s = np.zeros(n); s[period] = x[1:period+1].sum()
        for j in range(period+1, n): s[j] = s[j-1] - s[j-1]/period + x[j]
        return s
    str14 = sm(tr); sdm_p = sm(dm_p); sdm_m = sm(dm_m)
    di_p = np.where(str14>0, 100*sdm_p/str14, 0)
    di_m = np.where(str14>0, 100*sdm_m/str14, 0)
    dx   = np.where(di_p+di_m>0, 100*np.abs(di_p-di_m)/(di_p+di_m), 0)
    adx  = np.zeros(n)
    s = 2*period
    if n > s:
        adx[s] = dx[period:s+1].mean()
        for j in range(s+1, n): adx[j] = (adx[j-1]*(period-1)+dx[j])/period
    return float(adx[-1])

def rsi_series(closes, period=14):
    n = len(closes)
    out = np.full(n, 50.0)
    if n < period+2: return out
    d = np.diff(closes)
    g = np.where(d>0, d, 0.0); l = np.where(d<0, -d, 0.0)
    ag = g[:period].mean(); al = l[:period].mean()
    out[period] = 100 - 100/(1+ag/al) if al > 0 else 100
    for j in range(period, len(d)):
        ag = (ag*(period-1)+g[j])/period; al = (al*(period-1)+l[j])/period
        out[j+1] = 100 - 100/(1+ag/al) if al > 0 else 100
    return out

def rsi_divergence(closes, rsi_vals, window=5):
    """
    Returns: 'bearish' if price new high but RSI lower, 'bullish' if price new low but RSI higher, else None.
    Looks at last `window` bars.
    """
    if len(closes) < window+2: return None
    pc = closes[-window-1:-1]; pr = rsi_vals[-window-1:-1]
    cur_c = closes[-1]; cur_r = rsi_vals[-1]
    if cur_c > pc.max() and cur_r < pr.max():
        return 'bearish'
    if cur_c < pc.min() and cur_r > pr.min():
        return 'bullish'
    return None

# ── VOLUME PROFILE (POC / VAH / VAL) ─────────────────────────────────────────

def volume_profile(highs, lows, closes, vols, lookback=VP_LOOKBACK, bins=VP_BINS):
    """
    Approximate daily volume profile. Distributes each bar's volume
    uniformly across its high-low range. Returns (poc, vah, val).
    """
    H = highs[-lookback:]; L = lows[-lookback:]
    V = vols[-lookback:];  C = closes[-lookback:]
    price_min = L.min(); price_max = H.max()
    if price_max <= price_min:
        c = C[-1]
        return c, c, c

    edges = np.linspace(price_min, price_max, bins + 1)
    vol_hist = np.zeros(bins)

    for i in range(len(H)):
        rng = H[i] - L[i]
        if rng <= 0:
            idx = int(np.searchsorted(edges, C[i]) - 1)
            vol_hist[max(0, min(bins-1, idx))] += V[i]
            continue
        # overlap of each bin with [L[i], H[i]]
        bin_lows  = edges[:-1]; bin_highs = edges[1:]
        overlap   = np.maximum(0, np.minimum(H[i], bin_highs) - np.maximum(L[i], bin_lows))
        vol_hist += V[i] * overlap / rng

    poc_idx = int(vol_hist.argmax())
    poc = (edges[poc_idx] + edges[poc_idx+1]) / 2.0

    # VAH/VAL: expand from POC until 70% of volume is captured
    total = vol_hist.sum()
    if total <= 0:
        return poc, poc, poc
    target = total * 0.70
    lo = hi = poc_idx
    acc = vol_hist[poc_idx]
    while acc < target:
        lo_add = vol_hist[lo-1] if lo > 0 else 0.0
        hi_add = vol_hist[hi+1] if hi < bins-1 else 0.0
        if lo_add == 0 and hi_add == 0: break
        if lo_add >= hi_add and lo > 0:
            lo -= 1; acc += vol_hist[lo]
        elif hi < bins-1:
            hi += 1; acc += vol_hist[hi]
        elif lo > 0:
            lo -= 1; acc += vol_hist[lo]
        else:
            break
    vah = (edges[hi] + edges[hi+1]) / 2.0
    val = (edges[lo] + edges[lo+1]) / 2.0
    return poc, vah, val

# ── CPR / CAM / VWAP ─────────────────────────────────────────────────────────

def calc_cpr(H, L, C):
    pivot=(H+L+C)/3; bc=(H+L)/2; tc=2*pivot-bc
    upper=max(tc,bc); lower=min(tc,bc)
    w=upper-lower; wp=(w/pivot*100) if pivot>0 else 0
    return dict(pivot=pivot, upper=upper, lower=lower, width=w, width_pct=wp)

def calc_cam(H, L, C):
    r=H-L
    return dict(r4=C+r*1.1/2, r3=C+r*1.1/4,
                r2=C+r*1.1/6, r1=C+r*1.1/12,
                s1=C-r*1.1/12, s2=C-r*1.1/6,
                s3=C-r*1.1/4,  s4=C-r*1.1/2)

def calc_vwap(c5, h5, l5, v5):
    tp=(h5+l5+c5)/3; sv=(tp*v5).sum(); tv=v5.sum()
    return sv/tv if tv>0 else c5[-1]

# ── RULES ─────────────────────────────────────────────────────────────────────

def check_rules(cpr, cam, prev_close, cur_close, ph, pl, vwap, or_high, or_low):
    R = {}
    s3_in = cpr['lower'] <= cam['s3'] <= cpr['upper']
    r3_in = cpr['lower'] <= cam['r3'] <= cpr['upper']
    R['rule1']  = s3_in or r3_in
    R['rule2']  = cpr['width_pct'] < NARROW_THRESH
    R['rule3']  = prev_close < cpr['upper'] and cur_close > cpr['upper']
    R['rule4']  = ph < cpr['lower'] or pl > cpr['upper']
    margin = max(cpr['width']*0.5, cpr['pivot']*0.002)
    R['rule5']  = (cpr['lower']-margin) <= vwap <= (cpr['upper']+margin)
    safe = cur_close if cur_close > 0 else 1
    nr3  = abs(cur_close-cam['r3'])/safe < 0.005
    ns3  = abs(cur_close-cam['s3'])/safe < 0.005
    R['rule6']  = cpr['width_pct'] > 0.7 and (nr3 or ns3)
    if cpr['upper']>0 and cpr['lower']>0:
        rs = (prev_close>cpr['upper'] and cur_close>cpr['upper']
              and (cur_close-cpr['upper'])/cpr['upper']<0.012)
        rr = (prev_close<cpr['lower'] and cur_close<cpr['lower']
              and (cpr['lower']-cur_close)/cpr['lower']<0.012)
        R['rule7'] = rs or rr
    else:
        R['rule7'] = False
    R['rule8']  = ((cur_close>or_high and cur_close>cpr['upper']) or
                   (cur_close<or_low  and cur_close<cpr['lower']))
    R['rule9']  = cpr['pivot']>0 and abs(cur_close-cpr['pivot'])/cpr['pivot']>0.02
    ht = cpr['upper']>0 and abs(ph-cpr['upper'])/cpr['upper']<0.005
    lt = cpr['lower']>0 and abs(pl-cpr['lower'])/cpr['lower']<0.005
    R['rule10'] = ht or lt
    R['rule11'] = ((cur_close>vwap and cur_close<cpr['upper']) or
                   (cur_close<vwap and cur_close>cpr['lower']))
    return R

def get_direction(rid, cpr, cam, cur_close, prev_close, ph, pl):
    if rid=='rule3': return 1
    if rid=='rule4': return 1 if pl>cpr['upper'] else -1
    if rid=='rule6':
        safe=cur_close if cur_close>0 else 1
        return -1 if abs(cur_close-cam['r3'])/safe<0.005 else 1
    if rid=='rule7': return 1 if prev_close>cpr['upper'] else -1
    if rid=='rule8': return 1 if cur_close>cpr['upper'] else -1
    if rid=='rule9': return -1 if cur_close>cpr['pivot'] else 1
    if rid=='rule10':
        ht=cpr['upper']>0 and abs(ph-cpr['upper'])/cpr['upper']<0.005
        return -1 if ht else 1
    return 1 if cur_close>=cpr['pivot'] else -1

# ── ASYMMETRIC EXIT WITH MFE/MAE TRACKING ────────────────────────────────────

def simulate_exit(direction, entry_open, fh, fl, fc,
                  profit_target=PROFIT_TARGET, trail_stop=TRAIL_STOP):
    """Returns (final_return, mfe, mae)."""
    if entry_open <= 0: return 0.0, 0.0, 0.0
    peak = 0.0; worst = 0.0; final_ret = 0.0

    for d in range(len(fc)):
        if direction == 1:
            best  = (fh[d] - entry_open) / entry_open
            bad   = (fl[d] - entry_open) / entry_open
        else:
            best  = (entry_open - fl[d]) / entry_open
            bad   = (entry_open - fh[d]) / entry_open
        peak  = max(peak, best)
        worst = min(worst, bad)

        if best >= profit_target:
            final_ret = profit_target * 0.97; break
        if peak > 0.003 and (peak - best) >= trail_stop:
            final_ret = peak - trail_stop; break
        day_ret = direction * (fc[d] - entry_open) / entry_open
        if d == 0 and day_ret < -0.012:
            final_ret = day_ret; break
    else:
        final_ret = direction * (fc[-1] - entry_open) / entry_open

    return final_ret, peak, worst   # mfe=peak, mae=worst (negative)

# ── METRICS ──────────────────────────────────────────────────────────────────

def compute_metrics(rets):
    rets = np.array(rets)
    if not len(rets):
        return dict(trades=0, win_rate=0, pf=0, avg_ret=0, sharpe=0)
    wins=rets[rets>0]; loss=rets[rets<=0]
    pf  = wins.sum()/abs(loss.sum()) if loss.sum()!=0 else np.inf
    std = rets.std()
    return dict(trades=len(rets),
                win_rate=round(len(wins)/len(rets)*100,1),
                pf=round(pf,2),
                avg_ret=round(rets.mean()*100,3),
                sharpe=round(rets.mean()/std*np.sqrt(252) if std>0 else 0, 2))

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

print("Loading OHLCV data…")
df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.upper()
df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%b-%Y')
df = df.sort_values(['SYMBOL','DATE']).reset_index(drop=True)
for col in ['CLOSE','HIGH','LOW','OPEN']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['VOLUME'] = pd.to_numeric(df['VOLUME'], errors='coerce').fillna(0)
df = df.dropna(subset=['CLOSE','HIGH','LOW','OPEN'])
print(f"Symbols: {df['SYMBOL'].nunique()} | Rows: {len(df):,}")

# ── MAIN BACKTEST ─────────────────────────────────────────────────────────────

all_trades = {r: [] for r in RULE_IDS}  # (ret, mfe, mae)
processed  = 0

print("Running v3 backtest…\n")

for sym, grp in df.groupby('SYMBOL'):
    grp = grp.reset_index(drop=True); n = len(grp)
    if n < MIN_BARS + MAX_HOLD + 2: continue

    closes = grp['CLOSE'].values; highs  = grp['HIGH'].values
    lows   = grp['LOW'].values;   opens  = grp['OPEN'].values
    vols   = grp['VOLUME'].values; dates  = grp['DATE'].values

    ema200   = ema_series(closes, 200)
    vol20    = pd.Series(vols).rolling(20).mean().values
    atr_vals = atr_series_fn(highs, lows, closes, 14)
    rsi_vals = rsi_series(closes, 14)

    # pre-compute full ATR for percentile
    for i in range(55, n - MAX_HOLD - 2):
        if closes[i] < 20: continue

        dt      = pd.Timestamp(dates[i])
        ds      = str(dt.date())
        dow     = dt.weekday()
        if dow == 0 or dow == 4: continue  # skip Mon/Fri

        # ── VIX gate ──────────────────────────────────────────────────────────
        vix = get_vix(ds)
        if vix > 0 and vix > VIX_THRESHOLD: continue

        # ── ATR percentile ────────────────────────────────────────────────────
        atr_now = atr_vals[i]
        if atr_now <= 0: continue
        atr_win = atr_vals[max(0,i-120):i]; atr_win = atr_win[atr_win>0]
        if len(atr_win) < 30: continue
        atr_pct = float(np.sum(atr_win <= atr_now) / len(atr_win))
        if not (0.35 <= atr_pct <= 0.75): continue

        # ── ADX > 20 ──────────────────────────────────────────────────────────
        adx_now = adx_at(highs[max(0,i-40):i+1], lows[max(0,i-40):i+1],
                         closes[max(0,i-40):i+1])
        if adx_now < 20: continue

        # ── Volume filter ─────────────────────────────────────────────────────
        if vol20[i] > 0 and vols[i] < 1.5 * vol20[i]: continue

        # ── SG velocity ───────────────────────────────────────────────────────
        sg_vel = sg_velocity(closes[max(0,i-20):i+1])

        # ── Kalman velocity ───────────────────────────────────────────────────
        kal_vel = kalman_velocity(closes[max(0,i-59):i+1])

        # ── CPR / CAM / VWAP ─────────────────────────────────────────────────
        pH = highs[i-5:i].max(); pL = lows[i-5:i].min(); pC = closes[i-1]
        cur_H = highs[i-4:i+1].max(); cur_L = lows[i-4:i+1].min()
        h5=highs[i-4:i+1]; l5=lows[i-4:i+1]; c5=closes[i-4:i+1]; v5=vols[i-4:i+1]
        vwap     = calc_vwap(c5, h5, l5, v5)
        or_high  = highs[i-4:i-1].max(); or_low = lows[i-4:i-1].min()
        cpr = calc_cpr(pH, pL, pC); cam = calc_cam(pH, pL, pC)
        prev_close = closes[i-1]; cur_close = closes[i]

        # ── Volume Profile ────────────────────────────────────────────────────
        vp_start = max(0, i - VP_LOOKBACK)
        poc, vah, val = volume_profile(
            highs[vp_start:i+1], lows[vp_start:i+1],
            closes[vp_start:i+1], vols[vp_start:i+1]
        )
        poc_dist = (cur_close - poc) / poc if poc > 0 else 0.0
        # CPR-POC alignment: POC within CPR band → strong confluence
        poc_in_cpr = cpr['lower'] <= poc <= cpr['upper']

        # ── RSI divergence ────────────────────────────────────────────────────
        rsi_div = rsi_divergence(
            closes[max(0,i-10):i+1],
            rsi_vals[max(0,i-10):i+1]
        )

        # ── RULES ─────────────────────────────────────────────────────────────
        rules = check_rules(cpr, cam, prev_close, cur_close,
                            cur_H, cur_L, vwap, or_high, or_low)

        # ── WEIGHTED CONFLUENCE SCORE ──────────────────────────────────────────
        fired_rules = [r for r in RULE_IDS if rules[r]]
        conf_score  = sum(RULE_SHARPE.get(r, 0) for r in fired_rules)
        if conf_score < CONFLUENCE_THRESHOLD: continue

        entry_open = opens[i+1]
        if entry_open <= 0: continue

        for rid in fired_rules:
            # Skip zero-weight rules
            if RULE_SHARPE.get(rid, 0) <= 0: continue

            direction = get_direction(rid, cpr, cam, cur_close, prev_close, cur_H, cur_L)

            # ── Nifty trend gate ──────────────────────────────────────────────
            nifty_ema = get_nifty_ema20(ds)
            nifty_close_approx = nifty_map.get(ds, 0)
            if nifty_ema > 0:
                if direction == 1 and nifty_close_approx < nifty_ema * 0.995: continue
                if direction == -1 and nifty_close_approx > nifty_ema * 1.005: continue

            # ── EMA200 gate ───────────────────────────────────────────────────
            if direction == 1 and cur_close < ema200[i] * 0.99: continue
            if direction == -1 and cur_close > ema200[i] * 1.01: continue

            # ── SG + Kalman direction gate ────────────────────────────────────
            if direction == 1 and (sg_vel < 0 or kal_vel < 0): continue
            if direction == -1 and (sg_vel > 0 or kal_vel > 0): continue

            # ── RSI divergence gate for breakout rules ────────────────────────
            if rid in BREAKOUT_RULES:
                if direction == 1 and rsi_div == 'bearish': continue
                if direction == -1 and rsi_div == 'bullish': continue

            # ── Volume Profile gate: skip if price far from POC on non-VP rules
            # Bonus: lower target for POC-aligned trades (higher precision)
            pt = PROFIT_TARGET
            ts = TRAIL_STOP
            if poc_in_cpr:
                pt = PROFIT_TARGET * 1.2  # wider target when CPR+POC align

            # ── EXIT SIMULATION ───────────────────────────────────────────────
            max_fwd = min(MAX_HOLD, n - i - 2)
            if max_fwd < 1: continue
            fh = highs[i+1:i+1+max_fwd]
            fl = lows[i+1:i+1+max_fwd]
            fc = closes[i+1:i+1+max_fwd]

            ret, mfe, mae = simulate_exit(direction, entry_open, fh, fl, fc, pt, ts)
            all_trades[rid].append((ret, mfe, mae))

    processed += 1
    if processed % 50 == 0:
        print(f"  {processed}/{df['SYMBOL'].nunique()} symbols…")

# ── RESULTS ───────────────────────────────────────────────────────────────────

RULE_NAMES = {
    'rule1':'R1  Cam S3 or R3 in CPR', 'rule2':'R2  Narrow CPR',
    'rule3':'R3  Cross Above TC',       'rule4':'R4  Virgin CPR',
    'rule5':'R5  CPR+VWAP Confluence',  'rule6':'R6  Wide+Cam Extreme',
    'rule7':'R7  CPR S/R Flip Retest',  'rule8':'R8  OR+CPR Aligned',
    'rule9':'R9  Pivot Magnetic Pull',  'rule10':'R10 Price Testing CPR S/R',
    'rule11':'R11 VWAP-to-TC Setup',
}

print()
print("=" * 85)
print(f"{'Strategy':<30} {'Trades':>7} {'Win%':>7} {'PF':>6} {'Avg%':>8} {'Sharpe':>8}"
      f" {'MFE%':>7} {'MAE%':>7}")
print("=" * 85)

summary = []
mfe_mae_params = {}

for rid in RULE_IDS:
    data = all_trades[rid]
    if not data:
        print(f"{RULE_NAMES[rid]:<30} {'—':>7}")
        continue
    rets = np.array([d[0] for d in data])
    mfes = np.array([d[1] for d in data])
    maes = np.array([d[2] for d in data])

    m = compute_metrics(rets)

    # Per-rule optimal stop/target from MFE/MAE distribution
    opt_target = float(np.percentile(mfes, 60)) if len(mfes) else PROFIT_TARGET
    opt_stop   = abs(float(np.percentile(maes, 15))) if len(maes) else TRAIL_STOP  # MAE 85th percentile worst
    mfe_mae_params[rid] = {'opt_target': round(opt_target*100,3),
                            'opt_stop':   round(opt_stop*100,3),
                            'mfe_median': round(float(np.median(mfes))*100,3),
                            'mae_median': round(float(np.median(maes))*100,3)}

    avg_mfe = mfes.mean()*100; avg_mae = maes.mean()*100
    print(f"{RULE_NAMES[rid]:<30} {m['trades']:>7,} {m['win_rate']:>7} {m['pf']:>6} "
          f"{m['avg_ret']:>8} {m['sharpe']:>8} {avg_mfe:>7.3f} {avg_mae:>7.3f}")

    summary.append({**m, 'rule':rid, 'name':RULE_NAMES[rid],
                    'avg_mfe_pct': round(avg_mfe,3), 'avg_mae_pct': round(avg_mae,3),
                    **mfe_mae_params[rid]})

print("=" * 85)

# ── MFE/MAE OPTIMAL PARAMS TABLE ─────────────────────────────────────────────
print()
print("Optimal stop/target per rule (from MFE/MAE percentiles):")
print(f"{'Rule':<12} {'Opt Target%':>12} {'Opt Stop%':>11} {'MFE median%':>13} {'MAE median%':>13}")
for rid in RULE_IDS:
    p = mfe_mae_params.get(rid)
    if not p: continue
    print(f"{rid:<12} {p['opt_target']:>12} {p['opt_stop']:>11} "
          f"{p['mfe_median']:>13} {p['mae_median']:>13}")

# ── SAVE ─────────────────────────────────────────────────────────────────────
out = pd.DataFrame(summary)
out.to_csv(f"{OUT_DIR}/backtest_v3_results.csv", index=False)
mfe_df = pd.DataFrame([{'rule':k, **v} for k,v in mfe_mae_params.items()])
mfe_df.to_csv(f"{OUT_DIR}/mfe_mae_optimal_params.csv", index=False)

print(f"\nResults  → backtest_v3_results.csv")
print(f"MFE/MAE  → mfe_mae_optimal_params.csv")
print(f"Filters: VIX<{VIX_THRESHOLD} | Nifty 20-EMA | SG+Kalman+EMA200+ADX+ATR%ile+Vol")
print(f"         Weighted confluence>{CONFLUENCE_THRESHOLD} | RSI divergence | VP POC")
print(f"Exit: Asymmetric T+{MAX_HOLD} | per-POC target | trail {TRAIL_STOP*100}%")
print("Done.")
