"""
Comprehensive backtest — all 11 CPR rules individually.
Each rule evaluated standalone: quality filters kept (SG+Kalman+EMA200+ADX+ATR%+Vol+RSI div),
confluence filter REMOVED so R4/R9 get fair standalone assessment.
T1-cut = -1.2% (updated optimal). Exit: T+5 hold, 2.5% target, 0.8% trail.
"""
# [STANDALONE] One-off analysis script — not part of the ML training pipeline.


import pandas as pd
import numpy as np
import warnings
from scipy.signal import savgol_filter
warnings.filterwarnings('ignore')

DATA_FILE = r"C:\Users\drkkr\Downloads\NIFTY 500 OHLCV\ALL_SYMBOLS_OHLCV.csv"

MIN_BARS      = 60
MAX_HOLD      = 5
PROFIT_TARGET = 0.025
TRAIL_STOP    = 0.008
T1_CUT        = -0.012   # updated optimal
NARROW_THRESH = 0.5
VP_LOOKBACK   = 20
VP_BINS       = 60

RULE_IDS = [f'rule{i}' for i in range(1, 12)]
BREAKOUT_RULES = {'rule3', 'rule8'}

RULE_NAMES = {
    'rule1' : 'R1  Cam S3 OR R3 inside CPR band',
    'rule2' : 'R2  Narrow CPR  (<0.5% width)',
    'rule3' : 'R3  Price crosses above TC (breakout)',
    'rule4' : 'R4  Virgin CPR  (prev day outside band)',
    'rule5' : 'R5  CPR + VWAP confluence',
    'rule6' : 'R6  Wide CPR + Camarilla extreme touch',
    'rule7' : 'R7  CPR Support/Resistance flip retest',
    'rule8' : 'R8  Opening Range + CPR alignment',
    'rule9' : 'R9  Pivot magnetic pull (2%+ from pivot)',
    'rule10': 'R10 Price testing CPR S/R boundary',
    'rule11': 'R11 VWAP-to-TC setup',
}

# ── indicators ────────────────────────────────────────────────────────────────

def ema_series(arr, span):
    k = 2/(span+1); out = np.zeros(len(arr)); out[0] = arr[0]
    for i in range(1, len(arr)): out[i] = arr[i]*k + out[i-1]*(1-k)
    return out

def sg_velocity(closes, window=11, poly=3):
    if len(closes) < window: return 0.0
    return float(savgol_filter(closes, window, poly, deriv=1)[-1])

def kalman_velocity(closes, Q=1e-3, R=0.1):
    x = closes[0]; P = 1.0
    for z in closes: P += Q; K = P/(P+R); x = x+K*(z-x); P = (1-K)*P
    return x - closes[-2] if len(closes) > 1 else 0.0

def atr_series_fn(highs, lows, closes, period=14):
    n = len(closes); tr = np.zeros(n)
    for j in range(1, n):
        tr[j] = max(highs[j]-lows[j], abs(highs[j]-closes[j-1]), abs(lows[j]-closes[j-1]))
    out = np.zeros(n)
    if n > period:
        out[period] = tr[1:period+1].mean()
        for j in range(period+1, n): out[j] = (out[j-1]*(period-1)+tr[j])/period
    return out

def adx_at(highs, lows, closes, period=14):
    n = len(closes)
    if n < 2*period+2: return 0.0
    dm_p = np.zeros(n); dm_m = np.zeros(n); tr = np.zeros(n)
    for j in range(1, n):
        up = highs[j]-highs[j-1]; dn = lows[j-1]-lows[j]
        dm_p[j] = up if up > dn and up > 0 else 0
        dm_m[j] = dn if dn > up and dn > 0 else 0
        tr[j] = max(highs[j]-lows[j], abs(highs[j]-closes[j-1]), abs(lows[j]-closes[j-1]))
    def sm(x):
        s = np.zeros(n); s[period] = x[1:period+1].sum()
        for j in range(period+1, n): s[j] = s[j-1]-s[j-1]/period+x[j]
        return s
    str14 = sm(tr); sdm_p = sm(dm_p); sdm_m = sm(dm_m)
    di_p = np.where(str14>0, 100*sdm_p/str14, 0)
    di_m = np.where(str14>0, 100*sdm_m/str14, 0)
    dx   = np.where(di_p+di_m>0, 100*np.abs(di_p-di_m)/(di_p+di_m), 0)
    adx  = np.zeros(n); s = 2*period
    if n > s:
        adx[s] = dx[period:s+1].mean()
        for j in range(s+1, n): adx[j] = (adx[j-1]*(period-1)+dx[j])/period
    return float(adx[-1])

def rsi_series(closes, period=14):
    n = len(closes); out = np.full(n, 50.0)
    if n < period+2: return out
    d = np.diff(closes); g = np.where(d>0, d, 0.0); l = np.where(d<0, -d, 0.0)
    ag = g[:period].mean(); al = l[:period].mean()
    out[period] = 100-100/(1+ag/al) if al > 0 else 100
    for j in range(period, len(d)):
        ag = (ag*(period-1)+g[j])/period; al = (al*(period-1)+l[j])/period
        out[j+1] = 100-100/(1+ag/al) if al > 0 else 100
    return out

def rsi_divergence(closes, rsi_vals, window=5):
    if len(closes) < window+2: return None
    pc = closes[-window-1:-1]; pr = rsi_vals[-window-1:-1]
    cur_c = closes[-1]; cur_r = rsi_vals[-1]
    if cur_c > pc.max() and cur_r < pr.max(): return 'bearish'
    if cur_c < pc.min() and cur_r > pr.min(): return 'bullish'
    return None

def volume_profile_poc(highs, lows, closes, vols, lookback=VP_LOOKBACK, bins=VP_BINS):
    H = highs[-lookback:]; L = lows[-lookback:]; V = vols[-lookback:]; C = closes[-lookback:]
    pmin = L.min(); pmax = H.max()
    if pmax <= pmin: return C[-1], False
    edges = np.linspace(pmin, pmax, bins+1); vol_hist = np.zeros(bins)
    for i in range(len(H)):
        rng = H[i]-L[i]
        if rng <= 0:
            idx = int(np.searchsorted(edges, C[i])-1)
            vol_hist[max(0, min(bins-1, idx))] += V[i]; continue
        overlap = np.maximum(0, np.minimum(H[i], edges[1:]) - np.maximum(L[i], edges[:-1]))
        vol_hist += V[i]*overlap/rng
    poc_idx = int(vol_hist.argmax())
    poc = (edges[poc_idx]+edges[poc_idx+1])/2.0
    return poc, True

# ── levels ────────────────────────────────────────────────────────────────────

def calc_cpr(H, L, C):
    pivot=(H+L+C)/3; bc=(H+L)/2; tc=2*pivot-bc
    upper=max(tc,bc); lower=min(tc,bc); w=upper-lower
    return dict(pivot=pivot,upper=upper,lower=lower,width=w,
                width_pct=(w/pivot*100) if pivot>0 else 0)

def calc_cam(H, L, C):
    r=H-L
    return dict(r4=C+r*1.1/2,r3=C+r*1.1/4,r2=C+r*1.1/6,r1=C+r*1.1/12,
                s1=C-r*1.1/12,s2=C-r*1.1/6,s3=C-r*1.1/4,s4=C-r*1.1/2)

def calc_vwap(c5, h5, l5, v5):
    tp=(h5+l5+c5)/3; sv=(tp*v5).sum(); tv=v5.sum()
    return sv/tv if tv>0 else c5[-1]

# ── rules ─────────────────────────────────────────────────────────────────────

def check_rules(cpr, cam, prev_close, cur_close, ph, pl, vwap, or_high, or_low):
    R = {}
    R['rule1']  = (cpr['lower']<=cam['s3']<=cpr['upper']) or (cpr['lower']<=cam['r3']<=cpr['upper'])
    R['rule2']  = cpr['width_pct'] < NARROW_THRESH
    R['rule3']  = prev_close < cpr['upper'] and cur_close > cpr['upper']
    R['rule4']  = ph < cpr['lower'] or pl > cpr['upper']
    margin = max(cpr['width']*0.5, cpr['pivot']*0.002)
    R['rule5']  = (cpr['lower']-margin) <= vwap <= (cpr['upper']+margin)
    safe = cur_close if cur_close > 0 else 1
    nr3 = abs(cur_close-cam['r3'])/safe < 0.005
    ns3 = abs(cur_close-cam['s3'])/safe < 0.005
    R['rule6']  = cpr['width_pct'] > 0.7 and (nr3 or ns3)
    if cpr['upper'] > 0 and cpr['lower'] > 0:
        rs = (prev_close>cpr['upper'] and cur_close>cpr['upper']
              and (cur_close-cpr['upper'])/cpr['upper'] < 0.012)
        rr = (prev_close<cpr['lower'] and cur_close<cpr['lower']
              and (cpr['lower']-cur_close)/cpr['lower'] < 0.012)
        R['rule7'] = rs or rr
    else:
        R['rule7'] = False
    R['rule8']  = ((cur_close>or_high and cur_close>cpr['upper']) or
                   (cur_close<or_low  and cur_close<cpr['lower']))
    R['rule9']  = cpr['pivot'] > 0 and abs(cur_close-cpr['pivot'])/cpr['pivot'] > 0.02
    ht = cpr['upper'] > 0 and abs(ph-cpr['upper'])/cpr['upper'] < 0.005
    lt = cpr['lower'] > 0 and abs(pl-cpr['lower'])/cpr['lower'] < 0.005
    R['rule10'] = ht or lt
    R['rule11'] = ((cur_close>vwap and cur_close<cpr['upper']) or
                   (cur_close<vwap and cur_close>cpr['lower']))
    return R

def get_direction(rid, cpr, cam, cur_close, prev_close, ph, pl):
    if rid == 'rule3':  return 1
    if rid == 'rule4':  return 1 if pl > cpr['upper'] else -1
    if rid == 'rule6':
        safe = cur_close if cur_close > 0 else 1
        return -1 if abs(cur_close-cam['r3'])/safe < 0.005 else 1
    if rid == 'rule7':  return 1 if prev_close > cpr['upper'] else -1
    if rid == 'rule8':  return 1 if cur_close > cpr['upper'] else -1
    if rid == 'rule9':  return -1 if cur_close > cpr['pivot'] else 1
    if rid == 'rule10':
        return -1 if (cpr['upper']>0 and abs(ph-cpr['upper'])/cpr['upper']<0.005) else 1
    return 1 if cur_close >= cpr['pivot'] else -1

# ── exit ──────────────────────────────────────────────────────────────────────

def simulate(direction, entry_open, fh, fl, fc, pt):
    if entry_open <= 0: return 0.0, 0.0, 0.0, 'held'
    peak = 0.0; worst = 0.0
    for d in range(len(fc)):
        if direction == 1:
            best = (fh[d]-entry_open)/entry_open; bad = (fl[d]-entry_open)/entry_open
        else:
            best = (entry_open-fl[d])/entry_open; bad = (entry_open-fh[d])/entry_open
        peak = max(peak, best); worst = min(worst, bad)
        if best >= pt:
            return pt*0.97, peak, worst, 'target'
        if peak > 0.003 and (peak-best) >= TRAIL_STOP:
            return peak-TRAIL_STOP, peak, worst, 'trail'
        day_ret = direction*(fc[d]-entry_open)/entry_open
        if d == 0 and day_ret < T1_CUT:
            return day_ret, peak, worst, 't1cut'
    return direction*(fc[-1]-entry_open)/entry_open, peak, worst, 'held'

# ── metrics ───────────────────────────────────────────────────────────────────

def metrics(rets):
    rets = np.array(rets)
    if not len(rets):
        return dict(n=0,win=0,pf=0,avg=0,sharpe=0,kelly=0)
    wins = rets[rets>0]; loss = rets[rets<=0]
    wp   = len(wins)/len(rets)
    pf   = wins.sum()/abs(loss.sum()) if loss.sum() != 0 else 999
    std  = rets.std()
    avg_w = wins.mean() if len(wins) else 0
    avg_l = abs(loss.mean()) if len(loss) else 1
    kelly = wp - (1-wp)/( avg_w/avg_l) if avg_l > 0 else 0
    return dict(n=len(rets), win=round(wp*100,1), pf=round(pf,2),
                avg=round(rets.mean()*100,3),
                sharpe=round(rets.mean()/std*np.sqrt(252) if std>0 else 0, 2),
                kelly=round(kelly*100, 1))

# ── load ──────────────────────────────────────────────────────────────────────

print("Loading OHLCV…")
df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.upper()
df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%b-%Y')
df = df.sort_values(['SYMBOL','DATE']).reset_index(drop=True)
for col in ['CLOSE','HIGH','LOW','OPEN']: df[col] = pd.to_numeric(df[col], errors='coerce')
df['VOLUME'] = pd.to_numeric(df['VOLUME'], errors='coerce').fillna(0)
df = df.dropna(subset=['CLOSE','HIGH','LOW','OPEN'])
print(f"Symbols: {df['SYMBOL'].nunique()} | Rows: {len(df):,}")

# ── main loop ─────────────────────────────────────────────────────────────────

store = {r: {'rets':[], 'mfes':[], 'maes':[], 'reasons':[]} for r in RULE_IDS}
processed = 0

print("Running backtest — all 11 rules standalone…")

for sym, grp in df.groupby('SYMBOL'):
    grp = grp.reset_index(drop=True); n = len(grp)
    if n < MIN_BARS+MAX_HOLD+2: continue

    closes = grp['CLOSE'].values; highs  = grp['HIGH'].values
    lows   = grp['LOW'].values;   opens  = grp['OPEN'].values
    vols   = grp['VOLUME'].values

    ema200   = ema_series(closes, 200)
    vol20    = pd.Series(vols).rolling(20).mean().values
    atr_vals = atr_series_fn(highs, lows, closes, 14)
    rsi_vals = rsi_series(closes, 14)

    for i in range(55, n-MAX_HOLD-2):
        if closes[i] < 20: continue
        dt  = pd.Timestamp(grp['DATE'].iloc[i])
        dow = dt.weekday()
        if dow == 0 or dow == 4: continue

        # quality gates (same across all rules)
        atr_now = atr_vals[i]
        if atr_now <= 0: continue
        atr_win = atr_vals[max(0,i-120):i]; atr_win = atr_win[atr_win>0]
        if len(atr_win) < 30: continue
        atr_pct = float(np.sum(atr_win<=atr_now)/len(atr_win))
        if not (0.35 <= atr_pct <= 0.75): continue

        adx_now = adx_at(highs[max(0,i-40):i+1], lows[max(0,i-40):i+1],
                         closes[max(0,i-40):i+1])
        if adx_now < 20: continue
        if vol20[i] > 0 and vols[i] < 1.5*vol20[i]: continue

        sg_vel  = sg_velocity(closes[max(0,i-20):i+1])
        kal_vel = kalman_velocity(closes[max(0,i-59):i+1])

        pH = highs[i-5:i].max(); pL = lows[i-5:i].min(); pC = closes[i-1]
        cur_H = highs[i-4:i+1].max(); cur_L = lows[i-4:i+1].min()
        h5=highs[i-4:i+1]; l5=lows[i-4:i+1]; c5=closes[i-4:i+1]; v5=vols[i-4:i+1]
        vwap     = calc_vwap(c5, h5, l5, v5)
        or_high  = highs[i-4:i-1].max(); or_low = lows[i-4:i-1].min()
        cpr = calc_cpr(pH, pL, pC); cam = calc_cam(pH, pL, pC)
        prev_close = closes[i-1]; cur_close = closes[i]

        vp_start = max(0, i-VP_LOOKBACK)
        poc, _ = volume_profile_poc(highs[vp_start:i+1], lows[vp_start:i+1],
                                    closes[vp_start:i+1], vols[vp_start:i+1])
        poc_in_cpr = cpr['lower'] <= poc <= cpr['upper']
        pt = PROFIT_TARGET * (1.2 if poc_in_cpr else 1.0)

        rsi_div = rsi_divergence(closes[max(0,i-10):i+1], rsi_vals[max(0,i-10):i+1])

        rules = check_rules(cpr, cam, prev_close, cur_close, cur_H, cur_L,
                            vwap, or_high, or_low)
        entry_open = opens[i+1]
        if entry_open <= 0: continue

        for rid in RULE_IDS:
            if not rules[rid]: continue
            direction = get_direction(rid, cpr, cam, cur_close, prev_close, cur_H, cur_L)

            # EMA200 gate
            if direction == 1  and cur_close < ema200[i]*0.99: continue
            if direction == -1 and cur_close > ema200[i]*1.01: continue
            # SG + Kalman gate
            if direction == 1  and (sg_vel < 0 or kal_vel < 0): continue
            if direction == -1 and (sg_vel > 0 or kal_vel > 0): continue
            # RSI divergence gate (breakout rules only)
            if rid in BREAKOUT_RULES:
                if direction == 1  and rsi_div == 'bearish': continue
                if direction == -1 and rsi_div == 'bullish': continue

            max_fwd = min(MAX_HOLD, n-i-2)
            if max_fwd < 1: continue
            fh = highs[i+1:i+1+max_fwd]
            fl = lows [i+1:i+1+max_fwd]
            fc = closes[i+1:i+1+max_fwd]

            ret, mfe, mae, reason = simulate(direction, entry_open, fh, fl, fc, pt)
            store[rid]['rets'].append(ret)
            store[rid]['mfes'].append(mfe)
            store[rid]['maes'].append(mae)
            store[rid]['reasons'].append(reason)

    processed += 1
    if processed % 50 == 0:
        print(f"  {processed}/{df['SYMBOL'].nunique()} symbols…")

# ── REPORT ────────────────────────────────────────────────────────────────────

MFE_TH = [0.025, 0.03, 0.04, 0.05]

print()
print("=" * 130)
print("COMPREHENSIVE RESULTS — All 11 CPR Rules  |  T1-cut -1.2%  |  Target 2.5%  |  Trail 0.8%  |  Hold T+5")
print("Filters: SG+Kalman+EMA200+ADX+ATR-percentile+Volume+RSI-divergence  |  No confluence gate (standalone per rule)")
print("=" * 130)

hdr = (f"{'#':<3} {'Rule Description':<38} {'Trades':>7} {'Win%':>6} {'PF':>6} "
       f"{'Avg%':>7} {'Sharpe':>8} {'Kelly%':>8} "
       f"{'MFE%':>7} {'MAE%':>7} "
       f"{'MFE≥2.5%':>9} {'MFE≥3%':>7} {'MFE≥4%':>7} {'MFE≥5%':>7} "
       f"{'Target':>7} {'Trail':>6} {'T1Cut':>6} {'Held':>6}")
print(hdr)
print("-" * 130)

rows = []
for idx, rid in enumerate(RULE_IDS):
    s = store[rid]
    rets = np.array(s['rets']); mfes = np.array(s['mfes']); maes = np.array(s['maes'])
    reasons = s['reasons']
    n = len(rets)

    num = rid.replace('rule','R')
    name = RULE_NAMES[rid]

    if n == 0:
        print(f"{num:<3} {name:<38} {'No signals':>7}")
        rows.append({'rule':rid,'description':name,'trades':0})
        continue

    m = metrics(list(rets))

    avg_mfe = mfes.mean()*100; avg_mae = maes.mean()*100
    mfe_hits = [np.sum(mfes>=t)/n*100 for t in MFE_TH]

    tot = len(reasons)
    pt  = reasons.count('target')/tot*100
    ptr = reasons.count('trail')/tot*100
    pt1 = reasons.count('t1cut')/tot*100
    ph  = reasons.count('held')/tot*100

    # optimal stop/target from MFE/MAE percentiles
    opt_tgt  = float(np.percentile(mfes, 60))*100
    opt_stop = abs(float(np.percentile(maes, 15)))*100

    line = (f"{num:<3} {name:<38} {n:>7,} {m['win']:>6.1f} {m['pf']:>6.2f} "
            f"{m['avg']:>7.3f} {m['sharpe']:>8.2f} {m['kelly']:>8.1f} "
            f"{avg_mfe:>7.3f} {avg_mae:>7.3f} "
            f"{mfe_hits[0]:>9.1f} {mfe_hits[1]:>7.1f} {mfe_hits[2]:>7.1f} {mfe_hits[3]:>7.1f} "
            f"{pt:>6.1f}% {ptr:>5.1f}% {pt1:>5.1f}% {ph:>5.1f}%")
    print(line)

    rows.append({'rule':rid,'description':name,'trades':n,**m,
                 'avg_mfe_pct':round(avg_mfe,3),'avg_mae_pct':round(avg_mae,3),
                 'mfe_ge_2p5':round(mfe_hits[0],1),'mfe_ge_3':round(mfe_hits[1],1),
                 'mfe_ge_4':round(mfe_hits[2],1),'mfe_ge_5':round(mfe_hits[3],1),
                 'pct_target':round(pt,1),'pct_trail':round(ptr,1),
                 'pct_t1cut':round(pt1,1),'pct_held':round(ph,1),
                 'opt_target_pct':round(opt_tgt,3),'opt_stop_pct':round(opt_stop,3)})

print("=" * 130)
print()

# ── RANKED SUMMARY ────────────────────────────────────────────────────────────
print("RANKED BY SHARPE  (rules with trades only)")
print("-" * 90)
print(f"{'Rank':<5} {'Rule':<42} {'Trades':>7} {'Win%':>6} {'PF':>6} {'Avg%':>7} {'Sharpe':>8} {'Kelly%':>8}")
print("-" * 90)
ranked = sorted([r for r in rows if r.get('n',0)>0], key=lambda x: x.get('sharpe',0), reverse=True)
for rank, r in enumerate(ranked, 1):
    print(f"{rank:<5} {r['description']:<42} {r['n']:>7,} {r['win']:>6.1f} "
          f"{r['pf']:>6.2f} {r['avg']:>7.3f} {r['sharpe']:>8.2f} {r['kelly']:>8.1f}")

print()
print("OPTIMAL EXIT PARAMS PER RULE (MFE-60th / MAE-85th percentile)")
print("-" * 70)
print(f"{'Rule':<42} {'Opt Target%':>12} {'Opt Stop%':>11}")
print("-" * 70)
for r in rows:
    if r.get('n',0) == 0: continue
    print(f"{r['description']:<42} {r['opt_target_pct']:>12.3f} {r['opt_stop_pct']:>11.3f}")

out_df = pd.DataFrame(rows)
out_df.to_csv(r"D:\Claude code\nse-screener\comprehensive_results.csv", index=False)
print()
print("Saved → comprehensive_results.csv")
print(f"T1-cut used: {T1_CUT*100}%  |  Trail stop: {TRAIL_STOP*100}%  |  Base profit target: {PROFIT_TARGET*100}%")
