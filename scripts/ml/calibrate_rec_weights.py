"""
calibrate_rec_weights.py
Derives Buy/Sell recommendation engine weights from signal_dataset.csv.

Method
------
1. Construct binary/ordinal indicator columns that match the rec engine signals
2. Fit L2-penalised logistic regression with stratified 5-fold CV
   (penalty strength chosen by inner CV)
3. Compute SHAP-style mean |coefficient| ranking for interpretability
4. Bootstrap 200 resamples -> 95% CI on each weight
5. Scale coefficients to ±30-point system (max abs coeff -> 30 pts)
6. Write models/rec_weights.json (used by server + frontend)

Run
---
python scripts/ml/calibrate_rec_weights.py
"""

import os, json, time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

BASE       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIGNAL_CSV = os.path.join(BASE, 'models', 'signal_dataset.csv')
OUT_JSON   = os.path.join(BASE, 'models', 'rec_weights.json')

# ── Load ───────────────────────────────────────────────────────────────────────

def load_data():
    t0 = time.time()
    df = pd.read_csv(SIGNAL_CSV)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    print(f"  hit_t1 base rate: {df['hit_t1'].mean():.3f}")
    return df


# ── Feature engineering ────────────────────────────────────────────────────────

def build_features(df):
    """
    Map signal_dataset columns to binary/ordinal indicator features
    that correspond to the rec engine scoring signals.
    Direction convention: direction == 1 = LONG, direction == -1 = SHORT.
    """
    X = pd.DataFrame(index=df.index)

    d = df['direction'].values  # +1 or -1

    # ── VWAP alignment: price on correct side of VWAP for the trade direction
    # vwap_dist > 0 means price > VWAP
    X['vwap_aligned']  = ((df['vwap_dist'] * d) > 0).astype(float)
    X['vwap_opposed']  = ((df['vwap_dist'] * d) < 0).astype(float)

    # ── EMA200 alignment: price on correct side of 200-day EMA
    X['ema_aligned']   = ((df['ema200_dist'] * d) > 0).astype(float)
    X['ema_opposed']   = ((df['ema200_dist'] * d) < 0).astype(float)

    # ── CPR width: narrow CPR = better defined pivot, stronger signal
    cpr_med = df['cpr_width_pct'].median()
    X['narrow_cpr']    = (df['cpr_width_pct'] < cpr_med).astype(float)
    X['wide_cpr']      = (df['cpr_width_pct'] > cpr_med * 1.5).astype(float)

    # ── Momentum / extension
    mom_abs = df['mom5'].abs()
    X['not_extended']  = (mom_abs < 0.015).astype(float)
    X['mild_ext']      = ((mom_abs >= 0.015) & (mom_abs < 0.025)).astype(float)
    X['extended']      = (mom_abs >= 0.025).astype(float)

    # ── Momentum direction alignment
    X['mom_aligned']   = ((df['mom5'] * d) > 0).astype(float)
    X['mom_opposed']   = ((df['mom5'] * d) < 0).astype(float)

    # ── RSI regime
    X['rsi_favourable']= (
        ((d == 1) & (df['rsi14'] < 60) & (df['rsi14'] > 40)) |
        ((d == -1) & (df['rsi14'] > 40) & (df['rsi14'] < 60))
    ).astype(float)
    X['rsi_extreme']   = (
        ((d == 1) & (df['rsi14'] > 70)) |
        ((d == -1) & (df['rsi14'] < 30))
    ).astype(float)

    # ── Market VIX regime (India VIX)
    X['vix_low']       = (df['india_vix'] < 14).astype(float)   # calm, bullish bias
    X['vix_high']      = (df['india_vix'] > 18).astype(float)   # panic/volatile

    # ── Volume confirmation
    X['high_vol_rank'] = (df['vol_rank'] > 0.70).astype(float)
    X['low_vol_rank']  = (df['vol_rank'] < 0.30).astype(float)

    # ── Volume acceleration (surge)
    vol_acc_75 = df['vol_accel'].quantile(0.75)
    X['vol_surge']     = (df['vol_accel'] > vol_acc_75).astype(float)

    # ── Sector relative strength
    X['sector_rs_pos'] = (df['sector_rs_5d'] > 0).astype(float)
    X['sector_rs_neg'] = (df['sector_rs_5d'] < 0).astype(float)

    # ── Market relative strength (broad market momentum)
    X['market_rs_pos'] = (df['market_rs_5d'] > 0).astype(float)
    X['market_rs_neg'] = (df['market_rs_5d'] < 0).astype(float)

    # ── Number of concurrent rules fired (confluence)
    X['rules_3plus']   = (df['n_rules_fired'] >= 3).astype(float)
    X['rules_2']       = (df['n_rules_fired'] == 2).astype(float)
    X['rules_1']       = (df['n_rules_fired'] == 1).astype(float)

    # ── 52-week position
    X['near_hi52']     = (df['dist_hi52'] > -0.03).astype(float)  # within 3% of 52w high
    X['near_lo52']     = (df['dist_lo52'] < 0.03).astype(float)   # within 3% of 52w low

    # ── Delivery % (higher delivery = stronger conviction)
    if 'deliv_pct' in df.columns:
        deliv_75 = df['deliv_pct'].quantile(0.75)
        X['high_delivery'] = (df['deliv_pct'] > deliv_75).astype(float)

    # ── Put-Call ratio (sentiment)
    if 'pcr' in df.columns:
        pcr_med = df['pcr'].median()
        X['pcr_bearish']   = (df['pcr'] > pcr_med * 1.2).astype(float)  # high PCR = hedging

    # ── ATR rank (higher = more volatile = wider moves possible)
    X['atr_rank_high'] = (df['atr_pct_rank'] > 0.70).astype(float)

    # ── Day of week (Monday/Friday effects)
    X['monday']        = (df['dow'] == 0).astype(float)
    X['friday']        = (df['dow'] == 4).astype(float)

    # ── Sg velocity (signal velocity — momentum of momentum)
    X['sg_vel_pos']    = ((df['sg_vel'] * d) > 0).astype(float)

    # Drop any NaN rows
    X = X.fillna(0)
    return X


# ── Fit ────────────────────────────────────────────────────────────────────────

def fit_logistic(X, y):
    """
    Fit L2 logistic regression with nested 5-fold CV for penalty selection.
    Returns fitted pipeline and CV AUC.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Cs: penalty grid (larger C = less regularisation)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegressionCV(
            Cs=np.logspace(-3, 1, 20),
            cv=cv,
            scoring='roc_auc',
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
            n_jobs=1,
        ))
    ])
    pipe.fit(X, y)

    # OOF AUC
    y_prob = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        p = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegressionCV(
            Cs=[pipe['lr'].C_[0]], cv=3, scoring='roc_auc',
            penalty='l2', solver='lbfgs', max_iter=1000, random_state=42
        ))]); p.fit(X.iloc[tr], y.iloc[tr])
        y_prob[te] = p.predict_proba(X.iloc[te])[:, 1]
    oof_auc = roc_auc_score(y, y_prob)

    return pipe, oof_auc


# ── Bootstrap CI ───────────────────────────────────────────────────────────────

def bootstrap_coefs(X, y, pipe, n_boot=200, seed=0):
    rng    = np.random.default_rng(seed)
    coefs  = []
    lr     = pipe['lr']
    C_best = lr.C_[0]
    for _ in range(n_boot):
        idx = rng.integers(0, len(X), size=len(X))
        Xb  = X.iloc[idx].values
        yb  = y.iloc[idx].values
        sc  = StandardScaler().fit(Xb)
        from sklearn.linear_model import LogisticRegression
        m   = LogisticRegression(C=C_best, penalty='l2', solver='lbfgs',
                                 max_iter=1000, random_state=42)
        m.fit(sc.transform(Xb), yb)
        coefs.append(m.coef_[0])
    coefs = np.array(coefs)
    return coefs.mean(axis=0), coefs.std(axis=0), \
           np.percentile(coefs, 2.5, axis=0), np.percentile(coefs, 97.5, axis=0)


# ── Scale to points ────────────────────────────────────────────────────────────

def scale_to_points(coefs, target_max=30):
    """
    Linearly scale coefficients so max |coef| -> target_max points.
    Maintains sign and relative magnitude.
    """
    max_abs = np.max(np.abs(coefs))
    if max_abs == 0:
        return coefs
    return np.round(coefs / max_abs * target_max).astype(int)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Rec Engine Weight Calibration")
    print("=" * 60)

    df  = load_data()

    # Stratified subsample — 200k rows to avoid OOM on 1.4M x 33 parallel CV
    SAMPLE_N = 200_000
    if len(df) > SAMPLE_N:
        pos = df[df['hit_t1'] == 1].sample(SAMPLE_N // 2, random_state=42)
        neg = df[df['hit_t1'] == 0].sample(SAMPLE_N // 2, random_state=42)
        df  = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"  Subsampled to {len(df):,} (balanced 50/50)")

    X   = build_features(df)
    y   = df['hit_t1'].astype(int)

    feat_names = list(X.columns)
    print(f"\n  Features: {len(feat_names)}")
    print(f"  Positive rate: {y.mean():.3f}")

    print("\n  Fitting L2 logistic regression (nested 5-fold CV) ...")
    t0 = time.time()
    pipe, oof_auc = fit_logistic(X, y)
    print(f"  OOF AUC: {oof_auc:.4f}   ({time.time()-t0:.1f}s)")

    lr     = pipe['lr']
    coefs  = lr.coef_[0]
    C_best = float(lr.C_[0])
    print(f"  Best C (regularisation): {C_best:.4f}")

    print(f"\n  Bootstrapping 200 resamples for 95% CI ...")
    t1 = time.time()
    boot_mean, boot_std, ci_lo, ci_hi = bootstrap_coefs(X, y, pipe)
    print(f"  Done ({time.time()-t1:.1f}s)")

    # Scale to points
    pts     = scale_to_points(coefs, target_max=30)
    pts_lo  = scale_to_points(ci_lo,  target_max=30)
    pts_hi  = scale_to_points(ci_hi,  target_max=30)

    # Build output dict — sorted by absolute magnitude
    order = np.argsort(-np.abs(coefs))
    weights = {}
    print("\n  Signal weights (sorted by |effect|):")
    print(f"  {'Feature':<22}  {'Pts':>5}  {'95% CI':>14}  {'Coef':>8}")
    print("  " + "-" * 58)
    for i in order:
        fn = feat_names[i]
        weights[fn] = {
            'points':   int(pts[i]),
            'coef':     float(round(coefs[i], 5)),
            'ci_lo':    int(pts_lo[i]),
            'ci_hi':    int(pts_hi[i]),
            'boot_std': float(round(boot_std[i], 5)),
        }
        print(f"  {fn:<22}  {pts[i]:>+5}  [{pts_lo[i]:>+5}, {pts_hi[i]:>+5}]  {coefs[i]:>+8.4f}")

    out = {
        'version':    1,
        'oof_auc':    round(oof_auc, 5),
        'C_best':     C_best,
        'n_samples':  int(len(y)),
        'pos_rate':   float(round(y.mean(), 5)),
        'scale_max':  30,
        'weights':    weights,
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {OUT_JSON}")
    print(f"  OOF AUC: {oof_auc:.4f}")


if __name__ == '__main__':
    main()
