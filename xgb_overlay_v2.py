"""
XGBoost Overlay v2 — Fixes 1+2+3
Fix 1: XGBRegressor on actual return (not binary classifier)
Fix 2: Profit target 2.5%, trailing stop 0.8%, hold up to T+5
Fix 3: Min expected-return filter — predicted_return > 0.5%
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DATA_FILE = r"C:\Users\drkkr\Downloads\NIFTY 500 OHLCV\ALL_SYMBOLS_OHLCV.csv"
OUT_DIR   = r"D:\Claude code\nse-screener"

try:
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, r2_score
    import shap
except ImportError:
    print("pip install xgboost shap scikit-learn"); raise

# ── PARAMS (FIX 2) ────────────────────────────────────────────────────────────
PROFIT_TARGET = 0.025   # 2.5% (was 1.5%)
TRAIL_STOP    = 0.008   # 0.8% drawdown from peak (was 1.2%)
MAX_HOLD      = 5       # T+5 (was T+3)
MIN_PRED_RET  = 0.005   # Fix 3: only take if predicted return > 0.5%
MIN_BARS      = 60

# ── HELPERS ───────────────────────────────────────────────────────────────────

def ema(arr, span):
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values

def sg_velocity(closes, window=11, poly=3):
    from scipy.signal import savgol_filter
    if len(closes) < window: return 0.0
    return float(savgol_filter(closes, window, poly, deriv=1)[-1])

def rsi(closes, period=14):
    c = np.array(closes, dtype=float)
    if len(c) < period + 1: return 50.0
    d = np.diff(c)
    g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    ag = g[:period].mean(); al = l[:period].mean()
    for j in range(period, len(d)):
        ag = (ag*(period-1) + g[j]) / period
        al = (al*(period-1) + l[j]) / period
    return 100 - (100/(1 + ag/al)) if al > 0 else 100.0

def calc_cpr(H, L, C):
    pivot = (H+L+C)/3; bc = (H+L)/2; tc = 2*pivot - bc
    upper = max(tc, bc); lower = min(tc, bc)
    w = upper - lower; wp = (w/pivot*100) if pivot > 0 else 0
    return dict(pivot=pivot, upper=upper, lower=lower, width=w, width_pct=wp)

def calc_cam(H, L, C):
    r = H - L
    return dict(r3=C+r*1.1/4, s3=C-r*1.1/4)

def check_rules(cpr, cam, prev_close, cur_close, ph, pl, vwap, or_high, or_low):
    R = {}
    s3_in = cpr['lower'] <= cam['s3'] <= cpr['upper']
    r3_in = cpr['lower'] <= cam['r3'] <= cpr['upper']
    R['rule1'] = s3_in or r3_in
    R['rule2'] = cpr['width_pct'] < 0.5
    R['rule3'] = prev_close < cpr['upper'] and cur_close > cpr['upper']
    R['rule4'] = ph < cpr['lower'] or pl > cpr['upper']
    margin = max(cpr['width']*0.5, cpr['pivot']*0.002)
    R['rule5'] = (cpr['lower']-margin) <= vwap <= (cpr['upper']+margin)
    safe = cur_close if cur_close > 0 else 1
    nr3 = abs(cur_close-cam['r3'])/safe < 0.005
    ns3 = abs(cur_close-cam['s3'])/safe < 0.005
    R['rule6'] = cpr['width_pct'] > 0.7 and (nr3 or ns3)
    if cpr['upper'] > 0 and cpr['lower'] > 0:
        rs = (prev_close > cpr['upper'] and cur_close > cpr['upper']
              and (cur_close-cpr['upper'])/cpr['upper'] < 0.012)
        rr = (prev_close < cpr['lower'] and cur_close < cpr['lower']
              and (cpr['lower']-cur_close)/cpr['lower'] < 0.012)
        R['rule7'] = rs or rr
    else:
        R['rule7'] = False
    R['rule8']  = ((cur_close > or_high and cur_close > cpr['upper']) or
                   (cur_close < or_low  and cur_close < cpr['lower']))
    R['rule9']  = cpr['pivot'] > 0 and abs(cur_close-cpr['pivot'])/cpr['pivot'] > 0.02
    ht = cpr['upper'] > 0 and abs(ph-cpr['upper'])/cpr['upper'] < 0.005
    lt = cpr['lower'] > 0 and abs(pl-cpr['lower'])/cpr['lower'] < 0.005
    R['rule10'] = ht or lt
    R['rule11'] = ((cur_close > vwap and cur_close < cpr['upper']) or
                   (cur_close < vwap and cur_close > cpr['lower']))
    return R

def get_dir(rid, cpr, cam, cur_close, prev_close, ph, pl):
    if rid == 'rule3': return 1
    if rid == 'rule4': return 1 if pl > cpr['upper'] else -1
    if rid == 'rule6':
        safe = cur_close if cur_close > 0 else 1
        return -1 if abs(cur_close-cam['r3'])/safe < 0.005 else 1
    if rid == 'rule7': return 1 if prev_close > cpr['upper'] else -1
    if rid == 'rule8': return 1 if cur_close > cpr['upper'] else -1
    if rid == 'rule9': return -1 if cur_close > cpr['pivot'] else 1
    if rid == 'rule10':
        ht = cpr['upper'] > 0 and abs(ph-cpr['upper'])/cpr['upper'] < 0.005
        return -1 if ht else 1
    return 1 if cur_close >= cpr['pivot'] else -1

def asymmetric_exit(direction, entry_open, fh, fl, fc):
    """FIX 2: 2.5% target, 0.8% trail, T+5 hold."""
    if entry_open <= 0: return 0.0
    peak = 0.0
    for d in range(len(fc)):
        best  = (fh[d]-entry_open)/entry_open if direction == 1 else (entry_open-fl[d])/entry_open
        worst = (fl[d]-entry_open)/entry_open if direction == 1 else (entry_open-fh[d])/entry_open
        peak  = max(peak, best)
        if best >= PROFIT_TARGET:
            return PROFIT_TARGET * 0.97          # 3% slippage haircut
        if peak > 0.003 and (peak - best) >= TRAIL_STOP:
            return peak - TRAIL_STOP
        day_ret = direction * (fc[d]-entry_open) / entry_open
        if d == 0 and day_ret < -0.008:          # cut hard loss only >0.8% down
            return day_ret
    return direction * (fc[-1]-entry_open) / entry_open

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

RULE_IDS = [f'rule{i}' for i in range(1, 12)]

# ── BUILD FEATURE TABLE ───────────────────────────────────────────────────────

print("Building feature table with Fix 2 exits…")
records = []

for sym, grp in df.groupby('SYMBOL'):
    grp = grp.reset_index(drop=True); n = len(grp)
    if n < MIN_BARS + MAX_HOLD + 2: continue

    closes = grp['CLOSE'].values; highs = grp['HIGH'].values
    lows   = grp['LOW'].values;   opens = grp['OPEN'].values
    vols   = grp['VOLUME'].values; dates = grp['DATE'].values

    ema200 = ema(closes, 200)
    vol20  = pd.Series(vols).rolling(20).mean().values

    # pre-compute ATR series
    period = 14
    tr_arr = np.zeros(n)
    for j in range(1, n):
        tr_arr[j] = max(highs[j]-lows[j], abs(highs[j]-closes[j-1]), abs(lows[j]-closes[j-1]))
    atr_s = np.zeros(n)
    if n > period:
        atr_s[period] = tr_arr[1:period+1].mean()
        for j in range(period+1, n):
            atr_s[j] = (atr_s[j-1]*(period-1) + tr_arr[j]) / period

    for i in range(55, n - MAX_HOLD - 2):
        if closes[i] < 20: continue
        dt  = pd.Timestamp(dates[i])
        dow = dt.weekday()
        if dow == 0 or dow == 4: continue

        atr_now = atr_s[i]
        if atr_now <= 0: continue
        atr_win = atr_s[max(0, i-120):i]; atr_win = atr_win[atr_win > 0]
        if len(atr_win) < 30: continue
        atr_pct = float(np.sum(atr_win <= atr_now) / len(atr_win))
        if not (0.35 <= atr_pct <= 0.75): continue

        if vol20[i] > 0 and vols[i] < 1.5 * vol20[i]: continue

        pH = highs[i-5:i].max(); pL = lows[i-5:i].min(); pC = closes[i-1]
        cur_H = highs[i-4:i+1].max(); cur_L = lows[i-4:i+1].min()
        h5 = highs[i-4:i+1]; l5 = lows[i-4:i+1]
        c5 = closes[i-4:i+1]; v5 = vols[i-4:i+1]
        tp = (h5+l5+c5)/3; vwap = (tp*v5).sum()/v5.sum() if v5.sum() > 0 else c5[-1]
        or_high = highs[i-4:i-1].max(); or_low = lows[i-4:i-1].min()

        cpr = calc_cpr(pH, pL, pC); cam = calc_cam(pH, pL, pC)
        prev_close = closes[i-1]; cur_close = closes[i]

        rules  = check_rules(cpr, cam, prev_close, cur_close, cur_H, cur_L, vwap, or_high, or_low)
        n_fired = sum(1 for r in RULE_IDS if rules[r])
        if n_fired < 2: continue

        sg_vel    = sg_velocity(closes[max(0, i-20):i+1])
        rsi14     = rsi(closes[max(0, i-28):i+1])
        mom5      = float((closes[i]-closes[i-5])/closes[i-5]) if closes[i-5] > 0 else 0
        vwap_dist = float((cur_close-vwap)/vwap) if vwap > 0 else 0
        ema200_d  = float((cur_close-ema200[i])/ema200[i]) if ema200[i] > 0 else 0
        vol_rank  = float(vols[i]/vol20[i]) if vol20[i] > 0 else 1.0
        entry_open = opens[i+1]
        if entry_open <= 0: continue

        for rid in RULE_IDS:
            if not rules[rid]: continue
            direction = get_dir(rid, cpr, cam, cur_close, prev_close, cur_H, cur_L)

            max_fwd = min(MAX_HOLD, n-i-2)
            if max_fwd < 1: continue
            fh = highs[i+1:i+1+max_fwd]
            fl = lows[i+1:i+1+max_fwd]
            fc = closes[i+1:i+1+max_fwd]

            ret = asymmetric_exit(direction, entry_open, fh, fl, fc)

            records.append({
                'date': dt, 'symbol': sym, 'rule': rid,
                'direction': direction,
                'cpr_width_pct': cpr['width_pct'],
                'vwap_dist': vwap_dist,
                'atr_pct_rank': atr_pct,
                'vol_rank': vol_rank,
                'n_rules_fired': n_fired,
                'sg_vel': sg_vel,
                'ema200_dist': ema200_d,
                'rsi14': rsi14,
                'mom5': mom5,
                'dow': dow,
                'return': ret,
                'win': int(ret > 0)
            })

print(f"Feature table: {len(records):,} records")
feat_df = pd.DataFrame(records)
feat_df['rule_id'] = feat_df['rule'].str.extract(r'(\d+)').astype(int)

FEATURES = ['cpr_width_pct','vwap_dist','atr_pct_rank','vol_rank','n_rules_fired',
            'sg_vel','ema200_dist','rsi14','mom5','dow','rule_id','direction']

cut = pd.Timestamp('2025-01-01')
train_df = feat_df[feat_df['date'] < cut].copy()
test_df  = feat_df[feat_df['date'] >= cut].copy()

print(f"\nTrain: {len(train_df):,} | Test: {len(test_df):,}")
print(f"Train avg ret: {train_df['return'].mean()*100:.3f}%  win: {train_df['win'].mean()*100:.1f}%")
print(f"Test  avg ret: {test_df['return'].mean()*100:.3f}%   win: {test_df['win'].mean()*100:.1f}%")

# ── FIX 1: XGBRegressor on actual return ─────────────────────────────────────

X_train = train_df[FEATURES].fillna(0).values
y_train = train_df['return'].values          # continuous return, not 0/1
X_test  = test_df[FEATURES].fillna(0).values
y_test  = test_df['return'].values

reg = xgb.XGBRegressor(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,     # prevents overfitting on rare high-return trades
    random_state=42,
    eval_metric='mae'
)
reg.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False)

pred_test = reg.predict(X_test)
mae = mean_absolute_error(y_test, pred_test)
r2  = r2_score(y_test, pred_test)
print(f"\nRegressor MAE: {mae*100:.3f}%  R²: {r2:.4f}")

# ── FIX 3: min predicted return filter ───────────────────────────────────────

test_df = test_df.copy()
test_df['pred_ret'] = pred_test
test_df['pass']     = test_df['pred_ret'] >= MIN_PRED_RET

kept = test_df[test_df['pass']]
print(f"\nFix 3 — pred_ret ≥ {MIN_PRED_RET*100:.1f}%:")
print(f"  Kept: {len(kept):,} / {len(test_df):,} ({len(kept)/len(test_df)*100:.1f}%)")
print(f"  Win rate:  {kept['win'].mean()*100:.1f}%  (unfiltered: {test_df['win'].mean()*100:.1f}%)")
print(f"  Avg ret:   {kept['return'].mean()*100:.3f}%  (unfiltered: {test_df['return'].mean()*100:.3f}%)")

wins_k  = kept[kept['win']==1]['return']
loss_k  = kept[kept['win']==0]['return']
pf      = wins_k.sum() / abs(loss_k.sum()) if loss_k.sum() != 0 else float('inf')
sharpe  = kept['return'].mean() / kept['return'].std() * np.sqrt(252) if kept['return'].std() > 0 else 0
print(f"  Profit factor: {pf:.2f}")
print(f"  Sharpe:        {sharpe:.2f}")

print(f"\nPer-rule (Fix 3 filtered, OOS 2025+):")
print(f"{'Rule':<12} {'Kept':>6} {'Win%':>8} {'AvgRet%':>10} {'PF':>7} {'Sharpe':>8}")
for rid in RULE_IDS:
    sub = kept[kept['rule'] == rid]
    if len(sub) < 5: continue
    w  = sub[sub['win']==1]['return']; l = sub[sub['win']==0]['return']
    pf_r = w.sum() / abs(l.sum()) if l.sum() != 0 else float('inf')
    sh_r = sub['return'].mean() / sub['return'].std() * np.sqrt(252) if sub['return'].std() > 0 else 0
    print(f"{rid:<12} {len(sub):>6,} {sub['win'].mean()*100:>8.1f} "
          f"{sub['return'].mean()*100:>10.3f} {pf_r:>7.2f} {sh_r:>8.2f}")

# ── SHAP ──────────────────────────────────────────────────────────────────────

print("\nSHAP importances…")
explainer = shap.TreeExplainer(reg)
sv = explainer.shap_values(X_test[:5000])
fi = pd.DataFrame({'feature': FEATURES, 'shap': np.abs(sv).mean(axis=0)})
fi = fi.sort_values('shap', ascending=False)
print(fi.to_string(index=False))

# ── SAVE ──────────────────────────────────────────────────────────────────────

reg.save_model(f"{OUT_DIR}/xgb_regressor_v2.json")
test_df.to_csv(f"{OUT_DIR}/xgb_v2_signals.csv", index=False)
fi.to_csv(f"{OUT_DIR}/xgb_v2_shap.csv", index=False)
feat_df.to_csv(f"{OUT_DIR}/xgb_v2_features.csv", index=False)

print(f"\nModel   → xgb_regressor_v2.json")
print(f"Signals → xgb_v2_signals.csv")
print(f"SHAP    → xgb_v2_shap.csv")
print(f"\nParams: target={PROFIT_TARGET*100:.1f}% trail={TRAIL_STOP*100:.1f}% hold=T+{MAX_HOLD} min_pred={MIN_PRED_RET*100:.1f}%")
print("Done.")
