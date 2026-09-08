"""
cpr_phase2b_kernel.py — Phase 2b: LightGBM + Optuna HPO Signal Scorer

Reads from /kaggle/input/cpr-screener-phase2b-inputs/:
  signal_dataset.csv

Writes to /kaggle/working/:
  lgbm_scorer.txt        — LightGBM model (direct signal scorer)
  shap_weights.json      — Mean |SHAP| per feature (for weight derivation)
  phase2b_metrics.json   — AUC, CV scores, best Optuna params
"""

import os, sys, json, subprocess, warnings, time
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    import optuna
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'optuna'], check=False)
    import optuna

try:
    import lightgbm as lgb
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'lightgbm'], check=False)
    import lightgbm as lgb

try:
    import shap
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'shap'], check=False)
    import shap

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

INPUT = '/kaggle/input/cpr-screener-phase2b-inputs'
WORK  = '/kaggle/working'

FEATURE_COLS = [
    'cpr_width_pct', 'vwap_dist', 'atr_pct_rank', 'vol_rank',
    'n_rules_fired', 'sg_vel', 'ema200_dist', 'rsi14',
    'mom5', 'dow', 'rule_id', 'direction',
    'dist_hi52', 'dist_lo52', 'vol_accel',
    'market_rs_5d', 'market_rs_20d', 'sector_rs_5d', 'sector_rs_20d',
    'deliv_pct', 'pcr', 'india_vix', 'conf_vol',
    'rsi_dir', 'hi52_dir', 'cpr_compress', 'cpr_pos',
    'dist_r1', 'dist_s1', 'mom3', 'mom10', 'mom20',
    'rsi_div', 'vol_accel_delta', 'days_since_52hi', 'expiry_dist',
    'regime_stability', 'transition_risk',
    # Sprint 4: weekly CPR
    'weekly_cpr_first_break', 'weekly_price_above_wtc',
]  # 40 features

# Features that should flip sign for SELL direction
DIRECTIONAL_FEATURES = {
    'vwap_dist', 'ema200_dist', 'mom5', 'mom3', 'mom10', 'mom20',
    'market_rs_5d', 'market_rs_20d', 'sector_rs_5d', 'sector_rs_20d',
    'sg_vel', 'dist_r1', 'dist_s1', 'rsi_dir', 'hi52_dir',
    'rsi_div', 'dist_hi52', 'dist_lo52',
}

# Feature names after direction adjustment (no prefix change; values flipped in-place)
LGBM_FEATURE_NAMES = FEATURE_COLS  # same names, values adjusted


def load_data():
    import glob as _glob
    csv_path = os.path.join(INPUT, 'signal_dataset.csv')
    if not os.path.exists(csv_path):
        hits = _glob.glob(os.path.join(INPUT, '**', 'signal_dataset.csv'), recursive=True)
        if hits:
            csv_path = hits[0]
            print(f'  Found at {csv_path}')
        else:
            # Dataset version may still be processing — poll up to 10 min
            print(f'  signal_dataset.csv not at expected path. Polling /kaggle/input ...')
            print(f'  Contents of /kaggle/input: {os.listdir("/kaggle/input")}')
            deadline = time.time() + 600
            found = False
            while time.time() < deadline:
                time.sleep(30)
                hits = _glob.glob('/kaggle/input/**/signal_dataset.csv', recursive=True)
                if hits:
                    csv_path = hits[0]
                    print(f'  Found at {csv_path}')
                    found = True
                    break
                print(f'  Still waiting... /kaggle/input: {os.listdir("/kaggle/input")}')
            if not found:
                raise FileNotFoundError(f'signal_dataset.csv not found under {INPUT}')

    print(f'Loading {csv_path} ...')
    df = pd.read_csv(csv_path)
    print(f'  {len(df):,} rows loaded.')

    # Encode string columns
    for col in FEATURE_COLS + ['actual_return']:
        if col not in df.columns:
            df[col] = 0.0
        if df[col].dtype == object:
            extracted = df[col].astype(str).str.extract(r'(\d+)')[0]
            if extracted.notna().mean() > 0.5:
                df[col] = pd.to_numeric(extracted, errors='coerce').fillna(0)
            else:
                df[col] = df[col].astype('category').cat.codes.astype(float)

    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    # Target: hit_t1 preferred, fallback to actual_return > threshold
    if 'hit_t1' in df.columns and df['hit_t1'].notna().mean() > 0.5:
        target = df['hit_t1'].astype(int)
        print(f'  Target: hit_t1  (pos rate={target.mean():.3f})')
    else:
        threshold = float(df['actual_return'].quantile(0.6))
        target = (df['actual_return'] > threshold).astype(int)
        print(f'  Target: actual_return > {threshold:.4f}  (pos rate={target.mean():.3f})')

    # Direction adjustment: flip directional features for SELL
    direction_num = df['direction'].map(lambda x: -1.0 if str(x).upper() in ('SELL', '-1', '-1.0') else 1.0)
    df_feats = df[FEATURE_COLS].copy()
    for col in DIRECTIONAL_FEATURES:
        if col in df_feats.columns:
            df_feats[col] = df_feats[col] * direction_num

    X = df_feats.values.astype(np.float32)
    y = target.values.astype(np.int32)
    return X, y


def cv_auc(params, X, y, n_splits=5):
    """Time-series 5-fold CV AUC."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=LGBM_FEATURE_NAMES)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        model  = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        pred = model.predict(X_val)
        auc  = roc_auc_score(y_val, pred)
        aucs.append(auc)
        print(f'    Fold {fold+1}: AUC={auc:.4f}')
    mean_auc = float(np.mean(aucs))
    print(f'    Mean AUC: {mean_auc:.4f}')
    return mean_auc


def run_optuna(X, y, n_trials=40):
    print(f'\n  Optuna HPO ({n_trials} trials, 400k subsample) ...')
    sub_n   = min(400_000, len(X))
    rng     = np.random.RandomState(42)
    sub_idx = rng.choice(len(X), sub_n, replace=False)
    sub_idx.sort()  # keep chronological order for TimeSeriesSplit
    X_sub, y_sub = X[sub_idx], y[sub_idx]

    def objective(trial):
        params = {
            'objective':         'binary',
            'metric':            'auc',
            'verbosity':         -1,
            'boosting_type':     'gbdt',
            'num_leaves':        trial.suggest_int('num_leaves', 20, 200),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction':  trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq':      trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        return cv_auc(params, X_sub, y_sub, n_splits=5)

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f'  Best AUC: {study.best_value:.4f}')
    print(f'  Best params: {study.best_params}')
    return study.best_params, study.best_value


def final_train(X, y, best_params):
    print('\n  Final retrain on full dataset ...')
    params = {
        'objective':     'binary',
        'metric':        'auc',
        'verbosity':     -1,
        'boosting_type': 'gbdt',
        **best_params,
    }
    # Use last 15% as validation for early stopping
    n_val  = int(len(X) * 0.15)
    X_tr, X_val = X[:-n_val], X[-n_val:]
    y_tr, y_val = y[:-n_val], y[-n_val:]
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=LGBM_FEATURE_NAMES)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model  = lgb.train(
        params, dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(50, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )
    pred  = model.predict(X_val)
    final_auc = roc_auc_score(y_val, pred)
    print(f'  Final val AUC: {final_auc:.4f}')
    return model, final_auc


def compute_shap(model, X, feature_names):
    print('\n  Computing SHAP values (10k sample) ...')
    rng     = np.random.RandomState(42)
    n_shap  = min(10_000, len(X))
    idx     = rng.choice(len(X), n_shap, replace=False)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X[idx])
    mean_abs  = np.abs(shap_vals).mean(axis=0)
    importances = {feat: float(mean_abs[i])
                   for i, feat in enumerate(feature_names)}
    # Sort descending
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))
    print('  Top 10 features:')
    for feat, val in list(importances.items())[:10]:
        print(f'    {feat:<25} {val:.4f}')
    return importances


def main():
    print('=' * 60)
    print('  CPR Phase 2b: LightGBM HPO Signal Scorer')
    print('=' * 60)

    t0 = time.time()
    X, y = load_data()
    print(f'  Dataset: {X.shape[0]:,} rows x {X.shape[1]} features')

    best_params, best_cv_auc = run_optuna(X, y, n_trials=40)
    model, final_auc = final_train(X, y, best_params)
    shap_weights = compute_shap(model, X, LGBM_FEATURE_NAMES)

    # Save model
    model_path = os.path.join(WORK, 'lgbm_scorer.txt')
    model.save_model(model_path)
    print(f'\n  Saved → {model_path}')

    # Save SHAP weights
    shap_path = os.path.join(WORK, 'shap_weights.json')
    with open(shap_path, 'w') as f:
        json.dump(shap_weights, f, indent=2)
    print(f'  Saved → {shap_path}')

    # Save metrics
    metrics = {
        'best_cv_auc':    round(best_cv_auc, 4),
        'final_val_auc':  round(final_auc, 4),
        'n_rows':         int(X.shape[0]),
        'n_features':     int(X.shape[1]),
        'best_params':    best_params,
        'runtime_min':    round((time.time() - t0) / 60, 1),
    }
    metrics_path = os.path.join(WORK, 'phase2b_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'  Saved → {metrics_path}')

    print('\n' + '=' * 60)
    print(f'  Phase 2b complete! CV AUC={best_cv_auc:.4f}  Val AUC={final_auc:.4f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
