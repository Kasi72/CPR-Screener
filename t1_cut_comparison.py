"""
T1-cut threshold comparison: -0.5% vs -0.8% (current) vs -1.2%
Single data-collection pass, replay exits for each threshold.
"""
# [STANDALONE] One-off analysis script — not part of the ML training pipeline.


import pandas as pd
import numpy as np
import warnings
from scipy.signal import savgol_filter
warnings.filterwarnings('ignore')

DATA_FILE = r"C:\Users\drkkr\Downloads\NIFTY 500 OHLCV\ALL_SYMBOLS_OHLCV.csv"

MIN_BARS=60; MAX_HOLD=5; PROFIT_TARGET=0.025; TRAIL_STOP=0.008
NARROW_THRESH=0.5; CONFLUENCE_THRESHOLD=3.5; VP_LOOKBACK=20; VP_BINS=60

RULE_SHARPE={'rule1':2.86,'rule2':4.75,'rule3':0.44,'rule4':0.0,'rule5':3.24,
             'rule6':3.71,'rule7':6.19,'rule8':2.44,'rule9':0.0,'rule10':2.89,'rule11':3.90}
RULE_IDS=[f'rule{i}' for i in range(1,12)]
BREAKOUT_RULES={'rule3','rule8'}

T1_VARIANTS=[-0.005, -0.008, -0.012]   # tight | current | loose
T1_LABELS  =['T1=-0.5%','T1=-0.8%(cur)','T1=-1.2%']

# ── helpers (same as backtest_v3) ─────────────────────────────────────────────

def ema_series(arr,span):
    k=2/(span+1); out=np.zeros(len(arr)); out[0]=arr[0]
    for i in range(1,len(arr)): out[i]=arr[i]*k+out[i-1]*(1-k)
    return out

def sg_velocity(closes,window=11,poly=3):
    if len(closes)<window: return 0.0
    return float(savgol_filter(closes,window,poly,deriv=1)[-1])

def kalman_velocity(closes,Q=1e-3,R=0.1):
    x=closes[0]; P=1.0
    for z in closes: P+=Q; K=P/(P+R); x=x+K*(z-x); P=(1-K)*P
    return x-closes[-2] if len(closes)>1 else 0.0

def atr_series_fn(highs,lows,closes,period=14):
    n=len(closes); tr=np.zeros(n)
    for j in range(1,n): tr[j]=max(highs[j]-lows[j],abs(highs[j]-closes[j-1]),abs(lows[j]-closes[j-1]))
    out=np.zeros(n)
    if n>period:
        out[period]=tr[1:period+1].mean()
        for j in range(period+1,n): out[j]=(out[j-1]*(period-1)+tr[j])/period
    return out

def adx_at(highs,lows,closes,period=14):
    n=len(closes)
    if n<2*period+2: return 0.0
    dm_p=np.zeros(n); dm_m=np.zeros(n); tr=np.zeros(n)
    for j in range(1,n):
        up=highs[j]-highs[j-1]; dn=lows[j-1]-lows[j]
        dm_p[j]=up if up>dn and up>0 else 0
        dm_m[j]=dn if dn>up and dn>0 else 0
        tr[j]=max(highs[j]-lows[j],abs(highs[j]-closes[j-1]),abs(lows[j]-closes[j-1]))
    def sm(x):
        s=np.zeros(n); s[period]=x[1:period+1].sum()
        for j in range(period+1,n): s[j]=s[j-1]-s[j-1]/period+x[j]
        return s
    str14=sm(tr); sdm_p=sm(dm_p); sdm_m=sm(dm_m)
    di_p=np.where(str14>0,100*sdm_p/str14,0); di_m=np.where(str14>0,100*sdm_m/str14,0)
    dx=np.where(di_p+di_m>0,100*np.abs(di_p-di_m)/(di_p+di_m),0)
    adx=np.zeros(n); s=2*period
    if n>s:
        adx[s]=dx[period:s+1].mean()
        for j in range(s+1,n): adx[j]=(adx[j-1]*(period-1)+dx[j])/period
    return float(adx[-1])

def rsi_series(closes,period=14):
    n=len(closes); out=np.full(n,50.0)
    if n<period+2: return out
    d=np.diff(closes); g=np.where(d>0,d,0.0); l=np.where(d<0,-d,0.0)
    ag=g[:period].mean(); al=l[:period].mean()
    out[period]=100-100/(1+ag/al) if al>0 else 100
    for j in range(period,len(d)):
        ag=(ag*(period-1)+g[j])/period; al=(al*(period-1)+l[j])/period
        out[j+1]=100-100/(1+ag/al) if al>0 else 100
    return out

def rsi_divergence(closes,rsi_vals,window=5):
    if len(closes)<window+2: return None
    pc=closes[-window-1:-1]; pr=rsi_vals[-window-1:-1]
    cur_c=closes[-1]; cur_r=rsi_vals[-1]
    if cur_c>pc.max() and cur_r<pr.max(): return 'bearish'
    if cur_c<pc.min() and cur_r>pr.min(): return 'bullish'
    return None

def volume_profile_poc(highs,lows,closes,vols,lookback=VP_LOOKBACK,bins=VP_BINS):
    H=highs[-lookback:]; L=lows[-lookback:]; V=vols[-lookback:]; C=closes[-lookback:]
    price_min=L.min(); price_max=H.max()
    if price_max<=price_min: return C[-1], False
    edges=np.linspace(price_min,price_max,bins+1)
    vol_hist=np.zeros(bins)
    for i in range(len(H)):
        rng=H[i]-L[i]
        if rng<=0:
            idx=int(np.searchsorted(edges,C[i])-1)
            vol_hist[max(0,min(bins-1,idx))]+=V[i]; continue
        overlap=np.maximum(0,np.minimum(H[i],edges[1:])-np.maximum(L[i],edges[:-1]))
        vol_hist+=V[i]*overlap/rng
    poc_idx=int(vol_hist.argmax())
    poc=(edges[poc_idx]+edges[poc_idx+1])/2.0
    # check CPR — caller will determine this
    return poc, True

def calc_cpr(H,L,C):
    pivot=(H+L+C)/3; bc=(H+L)/2; tc=2*pivot-bc
    upper=max(tc,bc); lower=min(tc,bc); w=upper-lower
    return dict(pivot=pivot,upper=upper,lower=lower,width=w,width_pct=(w/pivot*100) if pivot>0 else 0)

def calc_cam(H,L,C):
    r=H-L
    return dict(r4=C+r*1.1/2,r3=C+r*1.1/4,r2=C+r*1.1/6,r1=C+r*1.1/12,
                s1=C-r*1.1/12,s2=C-r*1.1/6,s3=C-r*1.1/4,s4=C-r*1.1/2)

def calc_vwap(c5,h5,l5,v5):
    tp=(h5+l5+c5)/3; sv=(tp*v5).sum(); tv=v5.sum()
    return sv/tv if tv>0 else c5[-1]

def check_rules(cpr,cam,prev_close,cur_close,ph,pl,vwap,or_high,or_low):
    R={}
    R['rule1']=(cpr['lower']<=cam['s3']<=cpr['upper']) or (cpr['lower']<=cam['r3']<=cpr['upper'])
    R['rule2']=cpr['width_pct']<NARROW_THRESH
    R['rule3']=prev_close<cpr['upper'] and cur_close>cpr['upper']
    R['rule4']=ph<cpr['lower'] or pl>cpr['upper']
    margin=max(cpr['width']*0.5,cpr['pivot']*0.002)
    R['rule5']=(cpr['lower']-margin)<=vwap<=(cpr['upper']+margin)
    safe=cur_close if cur_close>0 else 1
    R['rule6']=cpr['width_pct']>0.7 and (abs(cur_close-cam['r3'])/safe<0.005 or abs(cur_close-cam['s3'])/safe<0.005)
    if cpr['upper']>0 and cpr['lower']>0:
        R['rule7']=((prev_close>cpr['upper'] and cur_close>cpr['upper'] and (cur_close-cpr['upper'])/cpr['upper']<0.012) or
                    (prev_close<cpr['lower'] and cur_close<cpr['lower'] and (cpr['lower']-cur_close)/cpr['lower']<0.012))
    else: R['rule7']=False
    R['rule8']=(cur_close>or_high and cur_close>cpr['upper']) or (cur_close<or_low and cur_close<cpr['lower'])
    R['rule9']=cpr['pivot']>0 and abs(cur_close-cpr['pivot'])/cpr['pivot']>0.02
    ht=cpr['upper']>0 and abs(ph-cpr['upper'])/cpr['upper']<0.005
    lt=cpr['lower']>0 and abs(pl-cpr['lower'])/cpr['lower']<0.005
    R['rule10']=ht or lt
    R['rule11']=((cur_close>vwap and cur_close<cpr['upper']) or (cur_close<vwap and cur_close>cpr['lower']))
    return R

def get_direction(rid,cpr,cam,cur_close,prev_close,ph,pl):
    if rid=='rule3': return 1
    if rid=='rule4': return 1 if pl>cpr['upper'] else -1
    if rid=='rule6':
        safe=cur_close if cur_close>0 else 1
        return -1 if abs(cur_close-cam['r3'])/safe<0.005 else 1
    if rid=='rule7': return 1 if prev_close>cpr['upper'] else -1
    if rid=='rule8': return 1 if cur_close>cpr['upper'] else -1
    if rid=='rule9': return -1 if cur_close>cpr['pivot'] else 1
    if rid=='rule10':
        return -1 if (cpr['upper']>0 and abs(ph-cpr['upper'])/cpr['upper']<0.005) else 1
    return 1 if cur_close>=cpr['pivot'] else -1

# ── collect raw forward bars per trade ────────────────────────────────────────

# Store: (direction, entry_open, fh_arr, fl_arr, fc_arr, poc_in_cpr)
raw_trades = {r: [] for r in RULE_IDS}

print("Loading data…")
df=pd.read_csv(DATA_FILE)
df.columns=df.columns.str.strip().str.upper()
df['DATE']=pd.to_datetime(df['DATE'],format='%d-%b-%Y')
df=df.sort_values(['SYMBOL','DATE']).reset_index(drop=True)
for col in ['CLOSE','HIGH','LOW','OPEN']: df[col]=pd.to_numeric(df[col],errors='coerce')
df['VOLUME']=pd.to_numeric(df['VOLUME'],errors='coerce').fillna(0)
df=df.dropna(subset=['CLOSE','HIGH','LOW','OPEN'])
print(f"Rows: {len(df):,}")

processed=0
print("Collecting trades…")
for sym, grp in df.groupby('SYMBOL'):
    grp=grp.reset_index(drop=True); n=len(grp)
    if n<MIN_BARS+MAX_HOLD+2: continue
    closes=grp['CLOSE'].values; highs=grp['HIGH'].values
    lows=grp['LOW'].values; opens=grp['OPEN'].values
    vols=grp['VOLUME'].values; dates=grp['DATE'].values
    ema200=ema_series(closes,200)
    vol20=pd.Series(vols).rolling(20).mean().values
    atr_vals=atr_series_fn(highs,lows,closes,14)
    rsi_vals=rsi_series(closes,14)

    for i in range(55,n-MAX_HOLD-2):
        if closes[i]<20: continue
        dt=pd.Timestamp(dates[i]); dow=dt.weekday()
        if dow==0 or dow==4: continue
        atr_now=atr_vals[i]
        if atr_now<=0: continue
        atr_win=atr_vals[max(0,i-120):i]; atr_win=atr_win[atr_win>0]
        if len(atr_win)<30: continue
        if not (0.35<=float(np.sum(atr_win<=atr_now)/len(atr_win))<=0.75): continue
        adx_now=adx_at(highs[max(0,i-40):i+1],lows[max(0,i-40):i+1],closes[max(0,i-40):i+1])
        if adx_now<20: continue
        if vol20[i]>0 and vols[i]<1.5*vol20[i]: continue
        sg_vel=sg_velocity(closes[max(0,i-20):i+1])
        kal_vel=kalman_velocity(closes[max(0,i-59):i+1])
        pH=highs[i-5:i].max(); pL=lows[i-5:i].min(); pC=closes[i-1]
        cur_H=highs[i-4:i+1].max(); cur_L=lows[i-4:i+1].min()
        h5=highs[i-4:i+1]; l5=lows[i-4:i+1]; c5=closes[i-4:i+1]; v5=vols[i-4:i+1]
        vwap=calc_vwap(c5,h5,l5,v5); or_high=highs[i-4:i-1].max(); or_low=lows[i-4:i-1].min()
        cpr=calc_cpr(pH,pL,pC); cam=calc_cam(pH,pL,pC)
        prev_close=closes[i-1]; cur_close=closes[i]
        vp_start=max(0,i-VP_LOOKBACK)
        poc,_=volume_profile_poc(highs[vp_start:i+1],lows[vp_start:i+1],
                                  closes[vp_start:i+1],vols[vp_start:i+1])
        poc_in_cpr=cpr['lower']<=poc<=cpr['upper']
        rsi_div=rsi_divergence(closes[max(0,i-10):i+1],rsi_vals[max(0,i-10):i+1])
        rules=check_rules(cpr,cam,prev_close,cur_close,cur_H,cur_L,vwap,or_high,or_low)
        fired=[r for r in RULE_IDS if rules[r]]
        if sum(RULE_SHARPE.get(r,0) for r in fired)<CONFLUENCE_THRESHOLD: continue
        entry_open=opens[i+1]
        if entry_open<=0: continue
        for rid in fired:
            if RULE_SHARPE.get(rid,0)<=0: continue
            direction=get_direction(rid,cpr,cam,cur_close,prev_close,cur_H,cur_L)
            if direction==1 and cur_close<ema200[i]*0.99: continue
            if direction==-1 and cur_close>ema200[i]*1.01: continue
            if direction==1 and (sg_vel<0 or kal_vel<0): continue
            if direction==-1 and (sg_vel>0 or kal_vel>0): continue
            if rid in BREAKOUT_RULES:
                if direction==1 and rsi_div=='bearish': continue
                if direction==-1 and rsi_div=='bullish': continue
            max_fwd=min(MAX_HOLD,n-i-2)
            if max_fwd<1: continue
            fh=highs[i+1:i+1+max_fwd].copy()
            fl=lows[i+1:i+1+max_fwd].copy()
            fc=closes[i+1:i+1+max_fwd].copy()
            raw_trades[rid].append((direction,entry_open,fh,fl,fc,poc_in_cpr))

    processed+=1
    if processed%50==0: print(f"  {processed}/469…")

# ── replay exits for each T1-cut variant ──────────────────────────────────────

def replay(direction, entry_open, fh, fl, fc, poc_in_cpr, t1_cut):
    if entry_open<=0: return 0.0, 'held'
    pt=PROFIT_TARGET*(1.2 if poc_in_cpr else 1.0)
    peak=0.0
    for d in range(len(fc)):
        if direction==1:
            best=(fh[d]-entry_open)/entry_open; bad=(fl[d]-entry_open)/entry_open
        else:
            best=(entry_open-fl[d])/entry_open; bad=(entry_open-fh[d])/entry_open
        peak=max(peak,best)
        if best>=pt:           return pt*0.97, 'target'
        if peak>0.003 and (peak-best)>=TRAIL_STOP: return peak-TRAIL_STOP, 'trail'
        day_ret=direction*(fc[d]-entry_open)/entry_open
        if d==0 and day_ret<t1_cut: return day_ret, 't1cut'
    return direction*(fc[-1]-entry_open)/entry_open, 'held'

def metrics(rets):
    rets=np.array(rets)
    if not len(rets): return dict(n=0,win=0,pf=0,avg=0,sharpe=0)
    wins=rets[rets>0]; loss=rets[rets<=0]
    pf=wins.sum()/abs(loss.sum()) if loss.sum()!=0 else 999
    std=rets.std()
    return dict(n=len(rets),
                win=round(len(wins)/len(rets)*100,1),
                pf=round(pf,2),
                avg=round(rets.mean()*100,3),
                sharpe=round(rets.mean()/std*np.sqrt(252) if std>0 else 0,2))

RULE_NAMES={
    'rule1':'R1  Cam S3|R3 in CPR','rule2':'R2  Narrow CPR',
    'rule3':'R3  Cross Above TC','rule5':'R5  CPR+VWAP Confluence',
    'rule6':'R6  Wide+Cam Extreme','rule7':'R7  CPR S/R Flip Retest',
    'rule8':'R8  OR+CPR Aligned','rule10':'R10 CPR S/R Test','rule11':'R11 VWAP-to-TC',
}

print("\n" + "="*110)
print("T1-CUT COMPARISON  |  Profit target 2.5% | Trail stop 0.8% | Hold T+5")
print("="*110)

all_summary = []

for t1, label in zip(T1_VARIANTS, T1_LABELS):
    print(f"\n── {label} ──────────────────────────────────────────────────────────────")
    print(f"{'Strategy':<26} {'N':>5}  {'Win%':>6} {'PF':>6} {'Avg%':>7} {'Sharpe':>8}  {'T1cut%':>7} {'Trail%':>7} {'Target%':>8} {'Held%':>6}")
    print("-"*110)
    rule_rows=[]
    for rid in RULE_IDS:
        trades=raw_trades[rid]
        if not trades: continue
        rets=[]; reasons=[]
        for (direction,entry_open,fh,fl,fc,poc_in_cpr) in trades:
            r,reason=replay(direction,entry_open,fh,fl,fc,poc_in_cpr,t1)
            rets.append(r); reasons.append(reason)
        m=metrics(rets); n=len(reasons)
        pt1=reasons.count('t1cut')/n*100; ptr=reasons.count('trail')/n*100
        ptgt=reasons.count('target')/n*100; pheld=reasons.count('held')/n*100
        name=RULE_NAMES.get(rid,rid)
        print(f"{name:<26} {m['n']:>5}  {m['win']:>6} {m['pf']:>6} {m['avg']:>7} {m['sharpe']:>8}  {pt1:>6.1f}% {ptr:>6.1f}% {ptgt:>7.1f}% {pheld:>5.1f}%")
        rule_rows.append({'variant':label,'rule':rid,'name':name,**m,
                          'pct_t1cut':round(pt1,1),'pct_trail':round(ptr,1),
                          'pct_target':round(ptgt,1),'pct_held':round(pheld,1)})
    all_summary.extend(rule_rows)

print("\n" + "="*110)
print("SUMMARY — best Sharpe per rule across 3 variants")
print("="*110)
print(f"{'Strategy':<26}  {T1_LABELS[0]:^22}  {T1_LABELS[1]:^22}  {T1_LABELS[2]:^22}")
print(f"{'':26}  {'Win%  PF  Sharpe':^22}  {'Win%  PF  Sharpe':^22}  {'Win%  PF  Sharpe':^22}")
print("-"*95)

summary_df=pd.DataFrame(all_summary)
for rid in RULE_IDS:
    sub=summary_df[summary_df['rule']==rid]
    if sub.empty: continue
    name=RULE_NAMES.get(rid,rid)
    vals=[]
    for lbl in T1_LABELS:
        row=sub[sub['variant']==lbl]
        if row.empty: vals.append(' '*22); continue
        r=row.iloc[0]
        vals.append(f"{r['win']:5.1f}%  {r['pf']:4.2f}  {r['sharpe']:6.2f}")
    print(f"{name:<26}  {vals[0]:^22}  {vals[1]:^22}  {vals[2]:^22}")

summary_df.to_csv(r"D:\Claude code\nse-screener\t1_cut_comparison.csv",index=False)
print("\nSaved → t1_cut_comparison.csv")
