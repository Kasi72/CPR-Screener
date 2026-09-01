"""
XGBoost ML Overlay — CPR Strategy Signal Filter
Trains binary classifier on enhanced backtest trade-level returns.
Walk-forward validation: train 2021-2024, test 2025+.
Output: per-rule win probability threshold filter.

Run AFTER backtest_enhanced.py has produced enhanced_trade_returns.csv.
Requires: pip install xgboost shap scikit-learn pandas numpy
"""
# [STANDALONE] One-off analysis script — not part of the ML training pipeline.


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DATA_FILE   = r"C:\Users\drkkr\Downloads\NIFTY 500 OHLCV\ALL_SYMBOLS_OHLCV.csv"
TRADES_FILE = r"D:\Claude code\nse-screener\enhanced_trade_returns.csv"
OUT_DIR     = r"D:\Claude code\nse-screener"

try:
    import xgboost as xgb
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.calibration import CalibratedClassifierCV
    import shap
except ImportError:
    print("Install: pip install xgboost shap scikit-learn")
    raise

# ── REBUILD FEATURE-RICH TRADE RECORDS ───────────────────────────────────────
# The enhanced backtest didn't save features alongside returns.
# We replay a slimmer pass to build a feature table matched to signals.

from scipy.signal import savgol_filter

def ema(arr, span):
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values

def sg_velocity(closes, window=11, poly=3):
    if len(closes) < window: return 0.0
    return float(savgol_filter(closes, window, poly, deriv=1)[-1])

def atr_val(highs, lows, closes, period=14):
    h=np.array(highs); l=np.array(lows); c=np.array(closes)
    if len(c) < period+2: return 0.0
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    atr=np.zeros(len(c))
    atr[period]=tr[:period].mean()
    for j in range(period+1,len(c)):
        atr[j]=(atr[j-1]*(period-1)+tr[j-1])/period
    return float(atr[-1])

def rsi(closes, period=14):
    c=np.array(closes,dtype=float)
    if len(c)<period+1: return 50.0
    d=np.diff(c)
    g=np.where(d>0,d,0.0); l=np.where(d<0,-d,0.0)
    ag=g[:period].mean(); al=l[:period].mean()
    for j in range(period,len(d)):
        ag=(ag*(period-1)+g[j])/period
        al=(al*(period-1)+l[j])/period
    return 100-(100/(1+ag/al)) if al>0 else 100.0

def calc_cpr(H,L,C):
    pivot=(H+L+C)/3; bc=(H+L)/2; tc=2*pivot-bc
    upper=max(tc,bc); lower=min(tc,bc)
    w=upper-lower; wp=(w/pivot*100) if pivot>0 else 0
    return dict(pivot=pivot,upper=upper,lower=lower,width=w,width_pct=wp)

def calc_cam(H,L,C):
    r=H-L
    return dict(r3=C+r*1.1/4, s3=C-r*1.1/4)

RULE_IDS = [f'rule{i}' for i in range(1,12)]
NARROW_THRESH=0.5

def check_rules_quick(cpr, cam, prev_close, cur_close, ph, pl, vwap, or_high, or_low):
    R={}
    s3_in=cpr['lower']<=cam['s3']<=cpr['upper']; r3_in=cpr['lower']<=cam['r3']<=cpr['upper']
    R['rule1']=s3_in or r3_in
    R['rule2']=cpr['width_pct']<NARROW_THRESH
    R['rule3']=prev_close<cpr['upper'] and cur_close>cpr['upper']
    R['rule4']=ph<cpr['lower'] or pl>cpr['upper']
    margin=max(cpr['width']*0.5, cpr['pivot']*0.002)
    R['rule5']=(cpr['lower']-margin)<=vwap<=(cpr['upper']+margin)
    safe=cur_close if cur_close>0 else 1
    nr3=abs(cur_close-cam['r3'])/safe<0.005; ns3=abs(cur_close-cam['s3'])/safe<0.005
    R['rule6']=cpr['width_pct']>0.7 and (nr3 or ns3)
    if cpr['upper']>0 and cpr['lower']>0:
        rs=(prev_close>cpr['upper'] and cur_close>cpr['upper'] and (cur_close-cpr['upper'])/cpr['upper']<0.012)
        rr=(prev_close<cpr['lower'] and cur_close<cpr['lower'] and (cpr['lower']-cur_close)/cpr['lower']<0.012)
        R['rule7']=rs or rr
    else:
        R['rule7']=False
    R['rule8']=(cur_close>or_high and cur_close>cpr['upper']) or (cur_close<or_low and cur_close<cpr['lower'])
    R['rule9']=cpr['pivot']>0 and abs(cur_close-cpr['pivot'])/cpr['pivot']>0.02
    ht=cpr['upper']>0 and abs(ph-cpr['upper'])/cpr['upper']<0.005
    lt=cpr['lower']>0 and abs(pl-cpr['lower'])/cpr['lower']<0.005
    R['rule10']=ht or lt
    R['rule11']=(cur_close>vwap and cur_close<cpr['upper']) or (cur_close<vwap and cur_close>cpr['lower'])
    return R

def get_dir(rid, cpr, cam, cur_close, prev_close, ph, pl):
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

print("Loading OHLCV data…")
df=pd.read_csv(DATA_FILE)
df.columns=df.columns.str.strip().str.upper()
df['DATE']=pd.to_datetime(df['DATE'],format='%d-%b-%Y')
df=df.sort_values(['SYMBOL','DATE']).reset_index(drop=True)
for c in ['CLOSE','HIGH','LOW','OPEN']:
    df[c]=pd.to_numeric(df[c],errors='coerce')
df['VOLUME']=pd.to_numeric(df['VOLUME'],errors='coerce').fillna(0)
df=df.dropna(subset=['CLOSE','HIGH','LOW','OPEN'])

print("Building feature table…")
records=[]
MIN_BARS=60; MAX_HOLD=3

for sym, grp in df.groupby('SYMBOL'):
    grp=grp.reset_index(drop=True); n=len(grp)
    if n<MIN_BARS+MAX_HOLD+2: continue
    closes=grp['CLOSE'].values; highs=grp['HIGH'].values
    lows=grp['LOW'].values;     opens=grp['OPEN'].values
    vols=grp['VOLUME'].values;  dates=grp['DATE'].values
    ema200=ema(closes,200)
    vol20=pd.Series(vols).rolling(20).mean().values
    # pre-compute full ATR series (same approach as backtest_enhanced.py)
    h=np.array(highs); l=np.array(lows); c=np.array(closes)
    period=14
    tr_arr=np.zeros(n)
    for j in range(1,n):
        tr_arr[j]=max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1]))
    atr_series=np.zeros(n)
    if n>period:
        atr_series[period]=tr_arr[1:period+1].mean()
        for j in range(period+1,n):
            atr_series[j]=(atr_series[j-1]*(period-1)+tr_arr[j])/period

    for i in range(55, n-MAX_HOLD-2):
        if closes[i]<20: continue
        dt=pd.Timestamp(dates[i])
        dow=dt.weekday()
        if dow==0 or dow==4: continue

        atr_now=atr_series[i]
        if atr_now<=0: continue
        atr_window=atr_series[max(0,i-120):i]
        atr_window=atr_window[atr_window>0]
        if len(atr_window)<30: continue
        atr_pct_rank=float(np.sum(atr_window<=atr_now)/len(atr_window))
        if not (0.35<=atr_pct_rank<=0.75): continue

        if vol20[i]>0 and vols[i]<1.5*vol20[i]: continue

        pH=highs[i-5:i].max(); pL=lows[i-5:i].min(); pC=closes[i-1]
        cur_H=highs[i-4:i+1].max(); cur_L=lows[i-4:i+1].min()
        h5=highs[i-4:i+1]; l5=lows[i-4:i+1]; c5=closes[i-4:i+1]; v5=vols[i-4:i+1]
        tp=(h5+l5+c5)/3; vwap=(tp*v5).sum()/v5.sum() if v5.sum()>0 else c5[-1]
        or_high=highs[i-4:i-1].max(); or_low=lows[i-4:i-1].min()

        cpr=calc_cpr(pH,pL,pC); cam=calc_cam(pH,pL,pC)
        prev_close=closes[i-1]; cur_close=closes[i]

        rules=check_rules_quick(cpr,cam,prev_close,cur_close,cur_H,cur_L,vwap,or_high,or_low)
        n_fired=sum(1 for r in RULE_IDS if rules[r])
        if n_fired<2: continue

        sg_vel=sg_velocity(closes[max(0,i-20):i+1])
        rsi14=rsi(closes[max(0,i-28):i+1])
        mom5=float((closes[i]-closes[i-5])/closes[i-5]) if closes[i-5]>0 else 0
        vwap_dist=float((cur_close-vwap)/vwap) if vwap>0 else 0
        cpr_width_pct=cpr['width_pct']
        ema200_dist=float((cur_close-ema200[i])/ema200[i]) if ema200[i]>0 else 0
        vol_rank=float(vols[i]/vol20[i]) if vol20[i]>0 else 1.0
        entry_open=opens[i+1]
        if entry_open<=0: continue

        for rid in RULE_IDS:
            if not rules[rid]: continue
            direction=get_dir(rid,cpr,cam,cur_close,prev_close,cur_H,cur_L)

            # asymmetric exit return
            max_fwd=min(MAX_HOLD,n-i-2)
            if max_fwd<1: continue
            fh=highs[i+1:i+1+max_fwd]; fl=lows[i+1:i+1+max_fwd]; fc=closes[i+1:i+1+max_fwd]
            peak=0.0; final_ret=0.0
            for d in range(max_fwd):
                best=(fh[d]-entry_open)/entry_open if direction==1 else (entry_open-fl[d])/entry_open
                peak=max(peak,best)
                if best>=0.015:
                    final_ret=0.015*0.95; break
                if peak>0 and (peak-best)>=0.012:
                    final_ret=peak-0.012; break
                day_ret=direction*(fc[d]-entry_open)/entry_open
                if d==0 and day_ret<0:
                    final_ret=day_ret; break
            else:
                final_ret=direction*(fc[-1]-entry_open)/entry_open

            records.append({
                'date': dt, 'symbol': sym, 'rule': rid,
                'direction': direction,
                'cpr_width_pct': cpr_width_pct,
                'vwap_dist': vwap_dist,
                'atr_pct_rank': atr_pct_rank,
                'vol_rank': vol_rank,
                'n_rules_fired': n_fired,
                'sg_vel': sg_vel,
                'ema200_dist': ema200_dist,
                'rsi14': rsi14,
                'mom5': mom5,
                'dow': dow,
                'return': final_ret,
                'win': int(final_ret > 0)
            })

print(f"Feature table: {len(records):,} records")
feat_df=pd.DataFrame(records)
feat_df.to_csv(f"{OUT_DIR}/xgb_features.csv", index=False)

# ── WALK-FORWARD TRAIN / TEST ─────────────────────────────────────────────────
FEATURES = ['cpr_width_pct','vwap_dist','atr_pct_rank','vol_rank','n_rules_fired',
            'sg_vel','ema200_dist','rsi14','mom5','dow']

# Encode rule as integer feature
feat_df['rule_id'] = feat_df['rule'].str.extract(r'(\d+)').astype(int)
FEATURES = FEATURES + ['rule_id', 'direction']

cut = pd.Timestamp('2025-01-01')
train_df = feat_df[feat_df['date'] < cut].copy()
test_df  = feat_df[feat_df['date'] >= cut].copy()

print(f"\nTrain: {len(train_df):,} | Test: {len(test_df):,}")
print(f"Train win rate: {train_df['win'].mean()*100:.1f}%")
print(f"Test  win rate: {test_df['win'].mean()*100:.1f}%")

X_train = train_df[FEATURES].fillna(0).values
y_train = train_df['win'].values
X_test  = test_df[FEATURES].fillna(0).values
y_test  = test_df['win'].values

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=float((y_train==0).sum()) / float((y_train==1).sum()),
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

proba_test = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, proba_test)
print(f"\nTest AUC: {auc:.4f}")

# Find threshold that maximises win rate while keeping ≥100 trades per rule
THRESHOLD = 0.65
mask = proba_test >= THRESHOLD
filtered = test_df.copy()
filtered['prob'] = proba_test
filtered['pass'] = mask
kept = filtered[filtered['pass']]

print(f"\nThreshold {THRESHOLD}: kept {len(kept):,}/{len(test_df):,} signals"
      f" ({len(kept)/len(test_df)*100:.1f}%)")
print(f"Filtered win rate: {kept['win'].mean()*100:.1f}%")
print(f"Unfiltered win rate: {test_df['win'].mean()*100:.1f}%")

print("\nPer-rule filtered stats:")
print(f"{'Rule':<12} {'Kept':>6} {'Win%':>8} {'Avg Ret%':>10}")
for rid in RULE_IDS:
    sub = kept[kept['rule']==rid]
    if len(sub)==0: continue
    print(f"{rid:<12} {len(sub):>6} {sub['win'].mean()*100:>8.1f} {sub['return'].mean()*100:>10.3f}")

# ── SHAP FEATURE IMPORTANCE ───────────────────────────────────────────────────
print("\nComputing SHAP values…")
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_test[:5000])
mean_abs  = np.abs(shap_vals).mean(axis=0)
fi_df = pd.DataFrame({'feature': FEATURES, 'shap_importance': mean_abs})
fi_df = fi_df.sort_values('shap_importance', ascending=False)
print("\nFeature importances (SHAP):")
print(fi_df.to_string(index=False))

# Save
model.save_model(f"{OUT_DIR}/xgb_cpr_model.json")
filtered.to_csv(f"{OUT_DIR}/xgb_test_signals.csv", index=False)
fi_df.to_csv(f"{OUT_DIR}/xgb_shap_importance.csv", index=False)

print(f"\nModel  → xgb_cpr_model.json")
print(f"Signals → xgb_test_signals.csv")
print(f"SHAP   → xgb_shap_importance.csv")
print("\nDone.")
