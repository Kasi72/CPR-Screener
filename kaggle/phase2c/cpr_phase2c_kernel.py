"""
cpr_phase2c_kernel.py — Phase 2c: Regime-Conditional LightGBM HPO Signal Scorer

Reads from /kaggle/input/cpr-screener-phase2c-inputs/:
  signal_dataset.csv   (output of build_dataset.py with Sprint 1 columns)

Outputs to /kaggle/working/:
  lgbm2c_global.txt        — global model (fallback)
  lgbm2c_regime_{0..3}.txt — per-regime models (states 0-3)
  shap_weights2c.json      — merged SHAP feature importances
  phase2c_metrics.json     — CV AUC, val AUC, per-regime stats
"""

import os, json, warnings, sys, subprocess, time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

try:
    import lightgbm as lgb
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'lightgbm'], check=False)
    import lightgbm as lgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'optuna'], check=False)
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import shap
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'shap'], check=False)
    import shap

warnings.filterwarnings('ignore')

INPUT = '/kaggle/input/cpr-screener-phase2c-inputs'
WORK  = '/kaggle/working'

# ── Feature sets ──────────────────────────────────────────────────────────────

BASE_FEATURES = [
    'cpr_width_pct', 'vwap_dist', 'atr_pct_rank', 'vol_rank',
    'n_rules_fired', 'sg_vel', 'ema200_dist', 'rsi14',
    'mom5', 'dow', 'rule_id', 'direction',
    'dist_hi52', 'dist_lo52', 'vol_accel',
    'market_rs_5d', 'market_rs_20d', 'sector_rs_5d', 'sector_rs_20d',
    'deliv_pct', 'pcr', 'india_vix',
    'conf_vol', 'rsi_dir', 'hi52_dir',
    'cpr_compress', 'cpr_pos', 'dist_r1', 'dist_s1',
    'mom3', 'mom10', 'mom20',
    'rsi_div', 'vol_accel_delta',
    'days_since_52hi', 'expiry_dist',
    # Sprint 1 new CPR features
    'cpr_overlap_pct', 'open_to_cpr_dist', 'prev_cpr_respected', 'cpr_zone_vol_ratio',
    # HMM regime as feature
    'hmm_regime',
    # Sprint 2A: compression/structure CPR features
    'open_inside_cpr', 'cpr_virgin', 'consecutive_narrow_cprs',
    'cpr_midpoint_trend', 'cpr_expansion_factor',
    # Sprint 2B: structural + context CPR features
    'cpr_above_prev_cpr', 'prev_close_inside_cpr', 'atr_to_cpr_ratio',
    'cpr_width_percentile_252d', 'prev_day_ochoa_type',
    # Sprint 3: gap + bar quality + volatility + volume structure
    'gap_pct', 'cpr_test_count_5d', 'prev_bar_close_pos',
    'atr_expansion', 'vol_trend_slope',
    # Sprint 4: weekly CPR features (rule15 primary signals)
    'weekly_cpr_first_break', 'weekly_price_above_wtc',
]  # 58 base features

INTERACTION_FEATURES = [
    'cpr_vol_interaction',      # cpr_compress × vol_rank
    'regime_momentum',          # hmm_regime × mom5
    'cpr_rsi_squeeze',          # (1 - cpr_width_pct) × rsi14
    'overlap_vol_signal',       # cpr_overlap_pct × cpr_zone_vol_ratio
    'rs_direction_alignment',   # (market_rs_5d + sector_rs_5d) × direction
    'virgin_momentum',          # cpr_virgin × mom5
    'narrow_breakout_vol',      # consecutive_narrow_cprs × vol_rank
]  # 7 interaction features

FEATURE_COLS = BASE_FEATURES + INTERACTION_FEATURES  # 63 total (56 base + 7 interactions)

# Features requiring direction-adjustment (sign flips for SELL signals)
DIRECTIONAL_FEATURES = {
    'dist_hi52', 'dist_lo52', 'vwap_dist', 'ema200_dist',
    'mom3', 'mom5', 'mom10', 'mom20',
    'market_rs_5d', 'market_rs_20d', 'sector_rs_5d', 'sector_rs_20d',
    'cpr_pos', 'dist_r1', 'dist_s1', 'sg_vel',
    'open_to_cpr_dist',
    'gap_pct',           # gap-up = good for long, gap-down = good for short
    # prev_bar_close_pos excluded: semantics ambiguous for shorts (near-high = resistance)
}

SUBSAMPLE_N  = 600_000
N_TRIALS     = 100
CV_SPLITS    = 5
EARLY_STOP   = 30
N_REGIMES    = 4
MIN_REGIME_N = 5_000   # min samples to train per-regime model
RANDOM_SEED  = 42


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    import glob as _glob
    # Search all of /kaggle/input/ — handles any dataset slug variation
    hits = _glob.glob('/kaggle/input/**/signal_dataset.csv', recursive=True)
    if hits:
        csv_path = hits[0]
        print(f"Found dataset at: {csv_path}")
    else:
        # Debug: show what IS mounted
        mounted = os.listdir('/kaggle/input') if os.path.exists('/kaggle/input') else []
        raise FileNotFoundError(
            f"signal_dataset.csv not found anywhere under /kaggle/input/. "
            f"Mounted sources: {mounted}"
        )
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  {len(df):,} rows, {df.columns.tolist()}")
    return df


def engineer_features(df):
    # Direction-adjust signed features for SELL signals
    df = df.copy()
    for col in DIRECTIONAL_FEATURES:
        if col in df.columns:
            df[col] = df[col] * df['direction'].fillna(1).astype(float)

    # Interaction features
    df['cpr_vol_interaction']   = (1 - df.get('cpr_compress', 0.5)) * df.get('vol_rank', 1.0)
    df['regime_momentum']       = df.get('hmm_regime', 0).clip(0, 3) * df.get('mom5', 0.0)
    df['cpr_rsi_squeeze']       = (1 - df.get('cpr_width_pct', 0.5).clip(0, 1)) * df.get('rsi14', 50.0) / 100
    df['overlap_vol_signal']    = df.get('cpr_overlap_pct', 0.5) * df.get('cpr_zone_vol_ratio', 1.0)
    df['rs_direction_alignment']= (df.get('market_rs_5d', 0.0) + df.get('sector_rs_5d', 0.0)) * df.get('direction', 1.0)
    df['virgin_momentum']       = df.get('cpr_virgin', 0.0) * df.get('mom5', 0.0)
    df['narrow_breakout_vol']   = df.get('consecutive_narrow_cprs', 0.0) * df.get('vol_rank', 1.0)

    # Fill missing feature columns with 0
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Encode string columns
    for col in FEATURE_COLS:
        if df[col].dtype == object:
            extracted = df[col].astype(str).str.extract(r'(\d+)')[0]
            if extracted.notna().mean() > 0.5:
                df[col] = pd.to_numeric(extracted, errors='coerce').fillna(0)
            else:
                df[col] = df[col].astype('category').cat.codes.astype(float)

    return df


def pick_target(df):
    """Auto-select best label: prefer win_rr if balanced, else hit_t1."""
    candidates = []
    for col in ['win_rr', 'hit_t3', 'hit_t1']:
        if col in df.columns:
            rate = df[col].mean()
            candidates.append((col, abs(rate - 0.35)))   # 35% positive rate is sweet-spot
    if not candidates:
        raise ValueError("No label column found (win_rr / hit_t3 / hit_t1).")
    best = min(candidates, key=lambda x: x[1])[0]
    print(f"  Target selected: {best}  (rate={df[best].mean():.1%})")
    return best


# ── Optuna HPO ────────────────────────────────────────────────────────────────

def hpo_objective(trial, X_tr, y_tr, n_splits=CV_SPLITS, pos_weight=None):
    params = {
        'objective':        'binary',
        'metric':           'auc',
        'verbosity':        -1,
        'boosting_type':    'gbdt',
        'n_estimators':     2000,
        'learning_rate':    trial.suggest_float('learning_rate', 1e-3, 0.15, log=True),
        'num_leaves':       trial.suggest_int('num_leaves', 20, 200),
        'max_depth':        trial.suggest_int('max_depth', 3, 10),
        'min_child_samples':trial.suggest_int('min_child_samples', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq':     trial.suggest_int('bagging_freq', 1, 7),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'random_state':     RANDOM_SEED,
    }
    if pos_weight is not None:
        params['scale_pos_weight'] = pos_weight
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    for tr_idx, va_idx in tscv.split(X_tr):
        X_t, X_v = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
        y_t, y_v = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(X_t, y_t,
                  eval_set=[(X_v, y_v)],
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                              lgb.log_evaluation(-1)])
        pred = model.predict_proba(X_v)[:, 1]
        aucs.append(roc_auc_score(y_v, pred))
    return float(np.mean(aucs))


def run_hpo(X_tr, y_tr, n_trials=N_TRIALS, label='global', pos_weight=None):
    print(f"  Optuna HPO ({n_trials} trials, {CV_SPLITS}-fold TS-CV) [{label}]"
          f"{f'  scale_pos_weight={pos_weight:.2f}' if pos_weight else ''}...")
    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(lambda t: hpo_objective(t, X_tr, y_tr, pos_weight=pos_weight),
                   n_trials=n_trials, show_progress_bar=False)
    print(f"  Best CV AUC ({label}): {study.best_value:.4f}")
    return study.best_params, study.best_value


def train_final(X_tr, y_tr, X_va, y_va, best_params):
    params = {
        'objective':     'binary',
        'metric':        'auc',
        'verbosity':     -1,
        'n_estimators':  3000,
        'random_state':  RANDOM_SEED,
        **best_params,
    }
    model = lgb.LGBMClassifier(**params)
    has_val = len(X_va) > 10 and len(y_va.unique()) > 1
    if has_val:
        model.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(200)])
        pred_va = model.predict_proba(X_va)[:, 1]
        val_auc = roc_auc_score(y_va, pred_va)
    else:
        # Regime-local fallback: use last 20% of training rows as validation
        cutoff = int(len(X_tr) * 0.80)
        X_tr_sub, X_fb = X_tr.iloc[:cutoff], X_tr.iloc[cutoff:]
        y_tr_sub, y_fb = y_tr.iloc[:cutoff], y_tr.iloc[cutoff:]
        fb_has_val = len(X_fb) > 10 and len(y_fb.unique()) > 1
        if fb_has_val:
            model.fit(X_tr_sub, y_tr_sub,
                      eval_set=[(X_fb, y_fb)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                  lgb.log_evaluation(200)])
            pred_fb = model.predict_proba(X_fb)[:, 1]
            val_auc = roc_auc_score(y_fb, pred_fb)
            print(f"    Val set empty — used last-20%% of train as val. Val AUC={val_auc:.4f}")
        else:
            n_est = min(params.get('n_estimators', 3000), 500)
            model.set_params(n_estimators=n_est)
            model.fit(X_tr, y_tr)
            val_auc = 0.0
            print("    Val set empty + regime too small — fixed 500 rounds.")
    print(f"    Val AUC: {val_auc:.4f}")
    return model, val_auc


def get_shap_weights(model, X_sample):
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample.iloc[:min(2000, len(X_sample))])
        if isinstance(sv, list):
            sv = sv[1]
        importance = np.abs(sv).mean(axis=0)
        total = importance.sum() or 1.0
        return {col: round(float(importance[i] / total), 6)
                for i, col in enumerate(X_sample.columns)}
    except Exception as e:
        print(f"    SHAP failed: {e} — using split importance.")
        imp = model.feature_importances_
        total = imp.sum() or 1.0
        return {col: round(float(imp[i] / total), 6)
                for i, col in enumerate(X_sample.columns)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CPR Phase 2c: Regime-Conditional LightGBM HPO")
    print("=" * 60)
    t0 = time.time()

    df = load_data()
    df = engineer_features(df)
    target = pick_target(df)

    # Temporal split: 70/15/15
    n    = len(df)
    n_tr = int(n * 0.70)
    n_va = int(n * 0.15)
    df_tr = df.iloc[:n_tr]
    df_va = df.iloc[n_tr:n_tr + n_va]
    df_te = df.iloc[n_tr + n_va:]
    print(f"  Train={len(df_tr):,}  Val={len(df_va):,}  Test={len(df_te):,}")

    # Subsample train for HPO (keep chronological order)
    if len(df_tr) > SUBSAMPLE_N:
        step = len(df_tr) // SUBSAMPLE_N
        df_hpo = df_tr.iloc[::max(step, 1)].iloc[:SUBSAMPLE_N]
    else:
        df_hpo = df_tr
    print(f"  HPO subsample: {len(df_hpo):,}")

    X_hpo   = df_hpo[FEATURE_COLS]
    y_hpo   = df_hpo[target]
    X_tr_f  = df_tr[FEATURE_COLS]
    y_tr_f  = df_tr[target]
    X_va    = df_va[FEATURE_COLS]
    y_va    = df_va[target]
    X_te    = df_te[FEATURE_COLS]
    y_te    = df_te[target]

    metrics = {
        'target': target,
        'n_rows': len(df),
        'n_features': len(FEATURE_COLS),
        'feature_cols': FEATURE_COLS,
        'regimes': {},
    }
    all_shap = {}

    # ── 1. Global model ────────────────────────────────────────────────────────
    print("\n[1/2] Global model HPO...")
    global_pos_weight = (y_tr_f == 0).sum() / max((y_tr_f == 1).sum(), 1)
    print(f"  Global class ratio (neg/pos): {global_pos_weight:.2f}")
    best_params, best_cv_auc = run_hpo(X_hpo, y_hpo, n_trials=N_TRIALS, label='global',
                                        pos_weight=round(global_pos_weight, 3))
    global_model, val_auc_global = train_final(X_tr_f, y_tr_f, X_va, y_va, best_params)

    # Test AUC
    pred_te = global_model.predict_proba(X_te)[:, 1]
    test_auc_global = roc_auc_score(y_te, pred_te) if len(y_te.unique()) > 1 else 0.0
    print(f"  Global test AUC: {test_auc_global:.4f}")

    out_global = os.path.join(WORK, 'lgbm2c_global.txt')
    global_model.booster_.save_model(out_global)
    print(f"  Saved -> {out_global}")

    shap_global = get_shap_weights(global_model, X_tr_f)
    all_shap['global'] = shap_global

    metrics.update({
        'best_cv_auc_global':  round(best_cv_auc, 4),
        'final_val_auc_global': round(val_auc_global, 4),
        'test_auc_global':     round(test_auc_global, 4),
        'best_params_global':  best_params,
    })

    # ── 2. Per-regime models ───────────────────────────────────────────────────
    print("\n[2/2] Per-regime models...")
    for regime in range(N_REGIMES):
        # All regime rows (chronologically sorted from df)
        mask_all = df['hmm_regime'] == regime
        df_r_all = df.loc[mask_all].reset_index(drop=True)

        n_regime_all = len(df_r_all)
        print(f"\n  Regime {regime}: {n_regime_all:,} total rows")

        if n_regime_all < MIN_REGIME_N:
            print(f"  Skip (< {MIN_REGIME_N}) — global model used for this regime.")
            metrics['regimes'][regime] = {'status': 'skipped', 'n_train': int(n_regime_all)}
            continue

        # Regime-local temporal split: 70/15/15 on regime's own timeline
        nr   = len(df_r_all)
        nr_tr = int(nr * 0.70)
        nr_va = int(nr * 0.15)
        df_rtr = df_r_all.iloc[:nr_tr]
        df_rva = df_r_all.iloc[nr_tr:nr_tr + nr_va]
        df_rte = df_r_all.iloc[nr_tr + nr_va:]
        print(f"    Regime-local split: train={len(df_rtr):,}  val={len(df_rva):,}  test={len(df_rte):,}")

        Xr_tr = df_rtr[FEATURE_COLS]
        yr_tr = df_rtr[target]
        Xr_va = df_rva[FEATURE_COLS]
        yr_va = df_rva[target]
        Xr_te = df_rte[FEATURE_COLS]
        yr_te = df_rte[target]

        # Per-regime scale_pos_weight
        r_pos_w = (yr_tr == 0).sum() / max((yr_tr == 1).sum(), 1)
        print(f"    Regime {regime} class ratio (neg/pos): {r_pos_w:.2f}")

        # Subsample for HPO
        if len(Xr_tr) > 150_000:
            step = len(Xr_tr) // 150_000
            Xr_hpo = Xr_tr.iloc[::max(step, 1)].iloc[:150_000]
            yr_hpo = yr_tr.iloc[::max(step, 1)].iloc[:150_000]
        else:
            Xr_hpo, yr_hpo = Xr_tr, yr_tr

        # More trials for small/hard regimes; 75 standard
        n_trials_r = 100 if n_regime_all < 20_000 else 75
        bp_r, cv_r = run_hpo(Xr_hpo, yr_hpo, n_trials=n_trials_r,
                              label=f'regime_{regime}', pos_weight=round(r_pos_w, 3))
        model_r, val_r = train_final(Xr_tr, yr_tr, Xr_va, yr_va, bp_r)

        te_auc_r = roc_auc_score(yr_te, model_r.predict_proba(Xr_te)[:, 1]) \
                   if (len(yr_te) > 10 and len(yr_te.unique()) > 1) else 0.0

        out_r = os.path.join(WORK, f'lgbm2c_regime_{regime}.txt')
        model_r.booster_.save_model(out_r)
        print(f"  Saved -> {out_r}")

        shap_r = get_shap_weights(model_r, Xr_tr)
        all_shap[f'regime_{regime}'] = shap_r

        metrics['regimes'][regime] = {
            'status':    'trained',
            'n_total':   int(n_regime_all),
            'n_train':   int(len(df_rtr)),
            'n_val':     int(len(df_rva)),
            'n_test':    int(len(df_rte)),
            'pos_weight': round(r_pos_w, 3),
            'cv_auc':    round(cv_r, 4),
            'val_auc':   round(val_r, 4),
            'test_auc':  round(te_auc_r, 4),
        }

    # ── Merge SHAP weights (global + regime blend) ────────────────────────────
    merged_shap = dict(shap_global)
    for key, weights in all_shap.items():
        if key == 'global':
            continue
        for feat, w in weights.items():
            merged_shap[feat] = round(merged_shap.get(feat, 0.0) * 0.5 + w * 0.5, 6)
    # Re-normalize
    total = sum(merged_shap.values()) or 1.0
    merged_shap = {k: round(v / total, 6) for k, v in merged_shap.items()}

    out_shap = os.path.join(WORK, 'shap_weights2c.json')
    with open(out_shap, 'w') as f:
        json.dump(merged_shap, f, indent=2)
    print(f"\nSaved SHAP -> {out_shap}")

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics['runtime_min'] = round((time.time() - t0) / 60, 2)
    out_metrics = os.path.join(WORK, 'phase2c_metrics.json')
    with open(out_metrics, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {out_metrics}")

    print("\n" + "=" * 60)
    print(f"  Phase 2c complete! Global val AUC={val_auc_global:.4f}  "
          f"({metrics['runtime_min']:.1f} min)")
    print("=" * 60)


if __name__ == '__main__':
    main()
