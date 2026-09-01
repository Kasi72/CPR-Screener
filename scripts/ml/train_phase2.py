"""
Phase 2: SHAP Dynamic Gate Weights + Conformal Prediction Calibration

Run AFTER Phase 1:
    python scripts/ml/train_phase2.py

Outputs:
    models/shap_gate_weights.json   — Per-gate importance weights from SHAP
    models/conformal_scores.json    — Calibration nonconformity scores (XGB + LGBM)
    models/phase2_metrics.json      — Metrics

Prerequisites:
    Phase 1 complete (hmm_params.json + lgbm_model.txt must exist)
"""

import os, json, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.ml.data_utils import MODELS_DIR, FEATURE_COLS, WIN_COL, MONOTONE_CONSTRAINTS, load_signal_dataset

import xgboost as xgb
import lightgbm as lgb
import shap


# Gate name → feature column mapping
# Each gate maps to the feature that represents it in the model
GATE_FEATURE_MAP = {
    'adx':       'atr_pct_rank',   # ADX and ATR both measure trend strength
    'atrPct':    'atr_pct_rank',
    'volSurge':  'vol_rank',
    'ema200':    'ema200_dist',
    'kalman':    'sg_vel',          # Kalman vel and SG vel are collinear
    'sgTrend':   'sg_vel',
    'rsiDiv':    'rsi14',
    'rsiEntry':  'rsi14',
    'niftyEma':  'mom5',            # Nifty trend proxy = momentum
    'vix':       'vol_rank',        # VIX correlates with volume
    'poc':       'vwap_dist',       # POC proximity ≈ VWAP distance
    'candleConf':'mom5',
    'volTrend':  'vol_rank',
}

BASE_GATE_WEIGHT = 1.0   # default weight when no SHAP data


def compute_shap_weights(xgb_model, lgbm_model, X, feature_names):
    """Compute combined gate weights from XGBoost native importance + LightGBM SHAP."""
    # XGBoost: use native feature importance (gain) — shape must match len(feature_names)
    print("  Computing XGBoost feature importance (gain)…")
    xgb_imp  = xgb_model.feature_importances_   # normalized gain, shape [n_features]
    shap_xgb = np.array(xgb_imp, dtype=np.float64)
    # Normalize to [0,1]
    if shap_xgb.max() > 0:
        shap_xgb = shap_xgb / shap_xgb.max()

    print("  Computing SHAP values for LightGBM…")
    try:
        explainer_lgb = shap.TreeExplainer(lgbm_model)
        sv            = explainer_lgb.shap_values(X)
        # Binary classifier returns list[2] (one array per class); index [1] for positive class
        shap_lgb      = np.abs(sv[1] if isinstance(sv, list) else sv).mean(axis=0)
        if shap_lgb.max() > 0:
            shap_lgb = shap_lgb / shap_lgb.max()
    except Exception as e:
        print(f"  LightGBM SHAP failed ({e}), using native importance.")
        imp = lgbm_model.feature_importance(importance_type='gain')
        shap_lgb = imp / (imp.max() or 1.0)

    # Guard against shape mismatch (e.g. LGBM trained on old feature set) — zero-fill rather than crash
    if len(shap_xgb) != len(feature_names):
        print(f"  WARN: XGB importance shape {len(shap_xgb)} != {len(feature_names)} features — zeroing")
        shap_xgb = np.zeros(len(feature_names))
    if len(shap_lgb) != len(feature_names):
        print(f"  WARN: LGBM SHAP shape {len(shap_lgb)} != {len(feature_names)} features — zeroing")
        shap_lgb = np.zeros(len(feature_names))

    # Average importance across both models
    combined = (shap_xgb + shap_lgb) / 2.0
    feat_importance = {feat: float(combined[i]) for i, feat in enumerate(feature_names)}

    # Normalize to [0.5, 2.0] range for gate weight scaling
    max_imp = max(feat_importance.values()) or 1.0
    gate_weights = {}
    for gate, feat in GATE_FEATURE_MAP.items():
        raw_imp = feat_importance.get(feat, 0.01)
        # Scale: gate weight = 0.5 + 1.5 * (importance / max_importance)
        gate_weights[gate] = round(0.5 + 1.5 * (raw_imp / max_imp), 3)

    print("  Gate weights from SHAP:")
    for g, w in sorted(gate_weights.items(), key=lambda x: -x[1]):
        feat = GATE_FEATURE_MAP.get(g, '?')
        imp  = feat_importance.get(feat, 0)
        print(f"    {g:<14} weight={w:.3f}  (feat={feat}, imp={imp:.4f})")

    return gate_weights, feat_importance


def calibrate_conformal(xgb_model, lgbm_model, X_cal, y_cal):
    """
    Compute nonconformity scores on the calibration set.
    Nonconformity score = 1 - P(true_class | x)
    Lower score = more conforming = more confident prediction.
    """
    print(f"  Calibrating on {len(X_cal)} held-out samples…")

    # XGBoost predictions (sklearn classifier — use predict_proba)
    xgb_prob = xgb_model.predict_proba(X_cal.astype(np.float32))[:, 1]
    xgb_prob = np.clip(xgb_prob, 0, 1)

    # LightGBM predictions
    lgbm_prob = lgbm_model.predict(X_cal)

    # Ensemble probability (simple average)
    ens_prob  = (xgb_prob + lgbm_prob) / 2.0

    # Nonconformity score: 1 - P(correct class)
    nc_scores_xgb  = 1 - np.where(y_cal == 1, xgb_prob,  1 - xgb_prob)
    nc_scores_lgbm = 1 - np.where(y_cal == 1, lgbm_prob, 1 - lgbm_prob)
    nc_scores_ens  = 1 - np.where(y_cal == 1, ens_prob,  1 - ens_prob)

    # Compute coverage at different alpha levels
    for alpha in [0.05, 0.10, 0.20]:
        q    = np.quantile(nc_scores_ens, 1 - alpha)
        cov  = (nc_scores_ens <= q).mean()
        print(f"    alpha={alpha:.2f}  q={q:.4f}  empirical coverage={cov:.1%}")

    return {
        'xgb':      sorted(nc_scores_xgb.tolist()),
        'lgbm':     sorted(nc_scores_lgbm.tolist()),
        'ensemble': sorted(nc_scores_ens.tolist()),
        'n_cal':    len(X_cal),
    }


def train_xgb_phase2(X_tr, y_tr, X_val=None, y_val=None):
    """Train a fresh XGBoost classifier on the current FEATURE_COLS (25 features)."""
    print("  Training XGBoost on current 25-feature dataset…")
    pos_weight = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric='auc',
        early_stopping_rounds=40,
        random_state=42,
        verbosity=0,
        monotone_constraints=tuple(MONOTONE_CONSTRAINTS),
    )
    eval_set = [(X_val, y_val)] if X_val is not None else None
    model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
    # Save refreshed model
    xgb_path = os.path.join(MODELS_DIR, 'xgb_phase2.json')
    model.save_model(xgb_path)
    print(f"  XGBoost saved → {xgb_path}")
    return model


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 60)
    print("  PHASE 2: SHAP Gate Weights + Conformal Calibration")
    print("=" * 60)

    lgbm_path = os.path.join(MODELS_DIR, 'lgbm_model.txt')
    if not os.path.exists(lgbm_path):
        raise FileNotFoundError(f"LightGBM model not found: {lgbm_path}\nRun Phase 1 first.")

    print("\n  Loading LightGBM model…")
    lgbm_model = lgb.Booster(model_file=lgbm_path)

    # Load signal dataset
    df  = load_signal_dataset()
    if WIN_COL not in df.columns:
        raise KeyError(f"WIN_COL='{WIN_COL}' not in dataset. Re-run build_dataset.py first.")
    # Fill any missing new feature cols (graceful for old datasets)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    X   = df[FEATURE_COLS].values.astype(np.float32)
    y   = df[WIN_COL].values.astype(int)

    # Split: 60% train, 20% calibration, 20% test
    n      = len(X)
    n_tr   = int(n * 0.60)
    n_cal  = int(n * 0.20)
    X_tr,  y_tr  = X[:n_tr],              y[:n_tr]
    X_cal, y_cal = X[n_tr:n_tr+n_cal],    y[n_tr:n_tr+n_cal]
    X_shap       = X[:min(n_tr, 5000)]    # subsample for SHAP speed

    # Train fresh XGBoost on 25-feature dataset (X_cal used as early-stop val set)
    xgb_model = train_xgb_phase2(X_tr, y_tr, X_val=X_cal, y_val=y_cal)

    # Phase 2A: SHAP weights
    print("\n── Phase 2A: SHAP Feature Importance → Gate Weights ────────────")
    gate_weights, feat_imp = compute_shap_weights(xgb_model, lgbm_model, X_shap, FEATURE_COLS)

    out_gw = os.path.join(MODELS_DIR, 'shap_gate_weights.json')
    with open(out_gw, 'w') as f:
        json.dump({
            'gate_weights':      gate_weights,
            'feature_importance': feat_imp,
            'updated_at':        pd.Timestamp.now().isoformat(),
        }, f, indent=2)
    print(f"  Saved → {out_gw}")

    # Phase 2B: Conformal calibration
    print("\n── Phase 2B: Conformal Prediction Calibration ──────────────────")
    conformal = calibrate_conformal(xgb_model, lgbm_model, X_cal, y_cal)

    out_cf = os.path.join(MODELS_DIR, 'conformal_scores.json')
    with open(out_cf, 'w') as f:
        json.dump(conformal, f)
    print(f"  Saved → {out_cf}  ({conformal['n_cal']} calibration scores)")

    metrics = {
        'top_gate':       max(gate_weights, key=gate_weights.get),
        'bot_gate':       min(gate_weights, key=gate_weights.get),
        'n_cal':          conformal['n_cal'],
        'q90_score':      round(float(np.quantile(conformal['ensemble'], 0.90)), 4),
    }
    with open(os.path.join(MODELS_DIR, 'phase2_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("  Phase 2 complete.")
    print("  Next: python scripts/ml/train_phase3.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
