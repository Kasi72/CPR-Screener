"""
Phase 1: HMM Market Regime Detector + LightGBM Signal Classifier

Run:
    python scripts/ml/train_phase1.py

Outputs:
    models/hmm_params.json          — Gaussian HMM parameters (inference in JS)
    models/lgbm_model.txt           — LightGBM booster (native format)
    models/phase1_metrics.json      — Training metrics

Prerequisites:
    pip install -r scripts/ml/requirements.txt
    python scripts/ml/build_dataset.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import sys
import lightgbm as lgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False
    print("  WARN: optuna not installed — using manual LightGBM params.")
    print("        pip install optuna  to enable HPO.")

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.ml.data_utils import (
    MODELS_DIR, FEATURE_COLS, WIN_COL, MONOTONE_CONSTRAINTS,
    build_nifty_regime_features, load_signal_dataset
)

REGIME_NAMES = ['Bull-Trend', 'Bear-Trend', 'Chop', 'High-Vol-Panic']

# ─────────────────────────── Phase 1A: HMM Regime ────────────────────────────

def train_hmm():
    print("\n── Phase 1A: HMM Market Regime ──────────────────────────────────")
    print("  Fetching Nifty 50 (^NSEI) 5-year daily data…")
    nifty = yf.download('^NSEI', period='5y', interval='1d', progress=False)
    if nifty.empty:
        raise RuntimeError("Failed to download Nifty 50 data from Yahoo Finance.")
    print(f"  Got {len(nifty)} daily bars.")

    obs = build_nifty_regime_features(nifty)
    print(f"  Observation matrix: {obs.shape}")

    # Standardise observations (HMM works better on z-scored features)
    scaler   = StandardScaler().fit(obs)
    obs_sc   = scaler.transform(obs)

    best_model, best_score = None, -np.inf
    for seed in [42, 7, 99]:
        model = GaussianHMM(
            n_components=4,
            covariance_type='diag',
            n_iter=200,
            random_state=seed,
            tol=1e-4
        )
        model.fit(obs_sc)
        score = model.score(obs_sc)
        print(f"  seed={seed}  log-likelihood={score:.2f}")
        if score > best_score:
            best_score, best_model = score, model

    # Decode current regime
    states = best_model.predict(obs_sc)
    counts = np.bincount(states, minlength=4)
    print(f"  State distribution: {dict(enumerate(counts.tolist()))}")

    # Map states to regime names by sorted mean return (state with highest mean return = Bull)
    mean_rets = [obs[states == s, 0].mean() for s in range(4)]
    order     = np.argsort(mean_rets)[::-1]   # highest return first
    regime_map = {int(order[0]): 'Bull-Trend',
                  int(order[1]): 'Bear-Trend',
                  int(order[2]): 'Chop',
                  int(order[3]): 'High-Vol-Panic'}
    current_state    = int(states[-1])
    current_regime   = regime_map[current_state]
    print(f"  Current regime: {current_regime} (state {current_state})")

    # Build date→state mapping and date→posterior for soft blending
    hmm_dates = nifty.index[-len(states):]
    state_by_date = {pd.Timestamp(d).strftime('%Y-%m-%d'): int(s)
                     for d, s in zip(hmm_dates, states)}

    # Posterior P(state | obs) for each date — used for soft regime blending
    posteriors = best_model.predict_proba(obs_sc)   # shape [T, 4]
    posterior_by_date = {pd.Timestamp(d).strftime('%Y-%m-%d'): posteriors[i].tolist()
                         for i, d in enumerate(hmm_dates)}

    # Save posteriors for soft blending at inference
    post_out = os.path.join(MODELS_DIR, 'hmm_posteriors.json')
    with open(post_out, 'w') as f:
        json.dump(posterior_by_date, f)
    print(f"  Posteriors saved → {post_out}")

    # Export HMM parameters for JS Viterbi inference
    params = {
        'n_components':  4,
        'startprob':     best_model.startprob_.tolist(),
        'transmat':      best_model.transmat_.tolist(),
        'means':         best_model.means_.tolist(),
        'covars':        best_model.covars_.tolist(),
        'scaler_mean':   scaler.mean_.tolist(),
        'scaler_scale':  scaler.scale_.tolist(),
        'regime_map':    {str(k): v for k, v in regime_map.items()},
        'current_regime': current_regime,
        'current_state':  current_state,
        'feature_names': ['ret', 'vol20', 'trend', 'vol_ratio', 'sg_vel'],
        'state_by_date': state_by_date,
    }
    out = os.path.join(MODELS_DIR, 'hmm_params.json')
    with open(out, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"  Saved → {out}")
    return current_regime, regime_map, state_by_date, posterior_by_date


# ──────────────────────────── Phase 1B: LightGBM ─────────────────────────────

def train_lgbm(current_regime, state_by_date=None, regime_map=None, posterior_by_date=None):
    print("\n── Phase 1B: LightGBM Signal Classifier ────────────────────────")
    df = load_signal_dataset()
    if WIN_COL not in df.columns:
        raise KeyError(f"WIN_COL='{WIN_COL}' not in dataset. Re-run build_dataset.py first.")
    print(f"  {len(df)} signals  |  target={WIN_COL}  |  win rate: {df[WIN_COL].mean():.1%}")

    # Assign HMM state to each signal (used for regime-specific sub-models)
    if state_by_date:
        df['hmm_state'] = df['date'].astype(str).str[:10].map(state_by_date).fillna(-1).astype(int)
    else:
        df['hmm_state'] = -1

    # --- Join regime_stability + transition_risk from posteriors ---
    if posterior_by_date:
        max_post_series = pd.Series({d: max(v) for d, v in posterior_by_date.items()})
        max_post_sorted = max_post_series.sort_index()
        max_post_prev   = max_post_sorted.shift(1).fillna(max_post_sorted.iloc[0])
        dates_str = df['date'].astype(str).str[:10]
        df['regime_stability'] = (
            dates_str.map(max_post_series) - dates_str.map(max_post_prev)
        ).fillna(0.0).astype(np.float32)
        df['transition_risk'] = (
            1.0 - dates_str.map(max_post_series)
        ).fillna(0.25).astype(np.float32)
        print(f"  Joined regime_stability + transition_risk ({df['regime_stability'].notna().sum()} rows)")
    else:
        df['regime_stability'] = 0.0
        df['transition_risk']  = 0.25

    # Fill any missing new feature cols (graceful for old datasets lacking Tier 2 features)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[WIN_COL].values.astype(int)

    # Temporal split: last 20% as test (preserve time order)
    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # Scale positive class weight for imbalanced labels
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    # ── Optuna HPO ───────────────────────────────────────────────────────────
    if OPTUNA_OK:
        print("  Running Optuna HPO (50 trials)…")

        def _objective(trial):
            p = {
                'objective':                    'binary',
                'metric':                       'auc',
                'verbose':                      -1,
                'seed':                         42,
                'bagging_freq':                 5,
                'scale_pos_weight':             pos_weight,
                'monotone_constraints':         MONOTONE_CONSTRAINTS,
                'monotone_constraints_method':  'advanced',
                'num_leaves':        trial.suggest_int('num_leaves', 15, 255),
                'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'feature_fraction':  trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 120),
                'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 5.0),
                'reg_lambda':        trial.suggest_float('reg_lambda', 0.0, 5.0),
                'max_depth':         trial.suggest_int('max_depth', 4, 12),
            }
            dtr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            dval_o = lgb.Dataset(X_te, label=y_te, reference=dtr, free_raw_data=False)
            m = lgb.train(p, dtr, num_boost_round=600,
                          valid_sets=[dval_o],
                          callbacks=[lgb.early_stopping(40, verbose=False)])
            return roc_auc_score(y_te, m.predict(X_te))

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(_objective, n_trials=50, show_progress_bar=False)
        best = study.best_params
        print(f"  Best params (AUC={study.best_value:.4f}): {best}")

        params = {
            'objective':                    'binary',
            'metric':                       ['binary_logloss', 'auc'],
            'verbose':                      -1,
            'seed':                         42,
            'bagging_freq':                 5,
            'scale_pos_weight':             pos_weight,
            'monotone_constraints':         MONOTONE_CONSTRAINTS,
            'monotone_constraints_method':  'advanced',
            **best,
        }
    else:
        params = {
            'objective':                    'binary',
            'metric':                       ['binary_logloss', 'auc'],
            'num_leaves':                   63,
            'learning_rate':                0.05,
            'feature_fraction':             0.8,
            'bagging_fraction':             0.8,
            'bagging_freq':                 5,
            'scale_pos_weight':             pos_weight,
            'min_child_samples':            20,
            'verbose':                      -1,
            'seed':                         42,
            'monotone_constraints':         MONOTONE_CONSTRAINTS,
            'monotone_constraints_method':  'advanced',
        }

    dtrain = lgb.Dataset(X_tr, label=y_tr,
                         feature_name=FEATURE_COLS,
                         free_raw_data=False)
    dval   = lgb.Dataset(X_te, label=y_te, reference=dtrain, free_raw_data=False)

    model = lgb.train(
        params, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(100)]
    )

    preds_prob = model.predict(X_te)
    preds_bin  = (preds_prob >= 0.5).astype(int)
    auc        = roc_auc_score(y_te, preds_prob)
    wr         = (preds_bin == y_te).mean()
    precision  = (preds_bin & y_te).sum() / max(preds_bin.sum(), 1)

    print(f"\n  Test AUC:        {auc:.4f}")
    print(f"  Test accuracy:   {wr:.1%}")
    print(f"  Signal precision:{precision:.1%}")
    print(classification_report(y_te, preds_bin, target_names=['Loss','Win']))

    # Feature importance
    imp = sorted(zip(FEATURE_COLS, model.feature_importance('gain').tolist()),
                 key=lambda x: -x[1])
    print("  Feature importance (gain):")
    for feat, gain in imp[:6]:
        print(f"    {feat:<18} {gain:.1f}")

    out = os.path.join(MODELS_DIR, 'lgbm_model.txt')
    model.save_model(out)
    print(f"\n  Saved → {out}")

    # ── Regime-aware sub-models with per-regime Optuna HPO ───────────────
    regime_aucs = {}
    if state_by_date and regime_map:
        print("\n── Phase 1B+: Regime-specific LightGBM sub-models (per-regime HPO) ──")
        for state in range(4):
            mask = df['hmm_state'] == state
            n_state = int(mask.sum())
            regime_name = regime_map.get(state, f'state{state}')
            if n_state < 500:
                print(f"  State {state} ({regime_name}): {n_state} signals — skip (< 500)")
                continue
            df_state = df.loc[mask].sort_values('date')  # chronological split
            Xs = df_state[FEATURE_COLS].values.astype(np.float32)
            ys = df_state[WIN_COL].values.astype(int)
            split_s  = int(len(Xs) * 0.8)
            X_tr_s, X_te_s = Xs[:split_s], Xs[split_s:]
            y_tr_s, y_te_s = ys[:split_s], ys[split_s:]
            pw_s = (y_tr_s == 0).sum() / max((y_tr_s == 1).sum(), 1)

            base_sub = {
                'objective':                   'binary',
                'metric':                      'auc',
                'verbose':                     -1,
                'seed':                        42,
                'bagging_freq':                5,
                'scale_pos_weight':            pw_s,
                'monotone_constraints':        MONOTONE_CONSTRAINTS,
                'monotone_constraints_method': 'advanced',
            }

            if OPTUNA_OK and n_state >= 2000:
                n_trials = 20
                print(f"  State {state} ({regime_name}): {n_state} signals — Optuna {n_trials} trials…")
                dtr_s_o = lgb.Dataset(X_tr_s, label=y_tr_s, free_raw_data=False)
                dte_s_o = lgb.Dataset(X_te_s, label=y_te_s, reference=dtr_s_o, free_raw_data=False)

                def _sub_obj(trial):
                    p = {
                        **base_sub,
                        'num_leaves':        trial.suggest_int('num_leaves', 15, 127),
                        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                        'feature_fraction':  trial.suggest_float('feature_fraction', 0.5, 1.0),
                        'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.5, 1.0),
                        'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
                        'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 3.0),
                        'reg_lambda':        trial.suggest_float('reg_lambda', 0.0, 3.0),
                        'max_depth':         trial.suggest_int('max_depth', 4, 10),
                    }
                    m = lgb.train(p, dtr_s_o, num_boost_round=400,
                                  valid_sets=[dte_s_o],
                                  callbacks=[lgb.early_stopping(30, verbose=False)])
                    return roc_auc_score(y_te_s, m.predict(X_te_s))

                sub_study = optuna.create_study(direction='maximize',
                                                sampler=optuna.samplers.TPESampler(seed=state))
                sub_study.optimize(_sub_obj, n_trials=n_trials, show_progress_bar=False)
                best_sub = sub_study.best_params
                print(f"    Best AUC={sub_study.best_value:.4f}  params={best_sub}")
                final_sub_params = {**base_sub, **best_sub}
            else:
                print(f"  State {state} ({regime_name}): {n_state} signals — fixed params…")
                final_sub_params = {**base_sub,
                                    'num_leaves': 63, 'learning_rate': 0.05,
                                    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
                                    'min_child_samples': 20, 'max_depth': 8}

            dtr_s_f = lgb.Dataset(X_tr_s, label=y_tr_s)
            m_s = lgb.train(final_sub_params, dtr_s_f, num_boost_round=400)
            auc_s = roc_auc_score(y_te_s, m_s.predict(X_te_s)) if len(np.unique(y_te_s)) > 1 else 0.5
            regime_aucs[regime_name] = round(float(auc_s), 4)
            out_s = os.path.join(MODELS_DIR, f'lgbm_regime_{state}.txt')
            m_s.save_model(out_s)
            print(f"  State {state} ({regime_name}): {n_state} signals  AUC={auc_s:.4f} → {out_s}")

        # ── Soft blending: save per-signal posterior weights for inference ──
        if posterior_by_date:
            print("\n  Computing soft-blend predictions on full dataset…")
            # Load all 4 sub-models
            sub_models = {}
            for state in range(4):
                p = os.path.join(MODELS_DIR, f'lgbm_regime_{state}.txt')
                if os.path.exists(p):
                    sub_models[state] = lgb.Booster(model_file=p)

            if len(sub_models) == 4:
                X_all  = df[FEATURE_COLS].values.astype(np.float32)
                y_all  = df[WIN_COL].values.astype(int)
                dates_all = df['date'].astype(str).str[:10]

                # Get posterior weights per signal
                posts = np.array([
                    posterior_by_date.get(d, [0.25, 0.25, 0.25, 0.25])
                    for d in dates_all
                ], dtype=np.float32)  # shape [N, 4]

                # Soft-blend prediction: sum_s P(s|obs) * P(win|x, s)
                preds_per_state = np.stack(
                    [sub_models[s].predict(X_all) for s in range(4)], axis=1
                )  # [N, 4]
                soft_preds = (posts * preds_per_state).sum(axis=1)

                split_oos = int(len(X_all) * 0.8)
                soft_auc = (roc_auc_score(y_all[split_oos:], soft_preds[split_oos:])
                            if len(np.unique(y_all[split_oos:])) > 1 else 0.5)
                print(f"  Soft-blend AUC (OOS holdout 20%): {soft_auc:.4f}")
                regime_aucs['soft_blend'] = round(float(soft_auc), 4)

                # Save soft blend config for predict_server
                soft_cfg = {'method': 'posterior_weighted', 'n_states': 4}
                with open(os.path.join(MODELS_DIR, 'soft_blend_config.json'), 'w') as f:
                    json.dump(soft_cfg, f)
                print("  Soft blend config saved → soft_blend_config.json")

    # ── Per-rule sub-models ─────────────────────────────────────────────────
    rule_aucs = {}
    print("\n── Phase 1B+++: Per-rule LightGBM sub-models ───────────────────────")
    # rule_id is already int-encoded (1..11) by load_signal_dataset
    for rule_num in range(1, 12):
        mask_r = df['rule_id'] == rule_num
        n_rule = int(mask_r.sum())
        if n_rule < 200:
            print(f"  rule{rule_num}: {n_rule} signals — skip (< 200)")
            continue
        df_rule = df.loc[mask_r].sort_values('date')
        Xr = df_rule[FEATURE_COLS].values.astype(np.float32)
        yr = df_rule[WIN_COL].values.astype(int)
        split_r = int(len(Xr) * 0.8)
        X_tr_r, X_te_r = Xr[:split_r], Xr[split_r:]
        y_tr_r, y_te_r = yr[:split_r], yr[split_r:]
        if len(np.unique(y_te_r)) < 2:
            print(f"  rule{rule_num}: {n_rule} signals — skip (uniform test labels)")
            continue
        pw_r = (y_tr_r == 0).sum() / max((y_tr_r == 1).sum(), 1)
        rule_params = {
            'objective': 'binary', 'metric': 'auc', 'verbose': -1,
            'seed': rule_num, 'bagging_freq': 5, 'scale_pos_weight': pw_r,
            'num_leaves': 31, 'learning_rate': 0.05,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8,
            'min_child_samples': 20, 'max_depth': 6,
        }
        dtr_r = lgb.Dataset(X_tr_r, label=y_tr_r)
        dte_r = lgb.Dataset(X_te_r, label=y_te_r, reference=dtr_r)
        m_r = lgb.train(rule_params, dtr_r, num_boost_round=300,
                        valid_sets=[dte_r],
                        callbacks=[lgb.early_stopping(30, verbose=False)])
        auc_r = roc_auc_score(y_te_r, m_r.predict(X_te_r))
        rule_aucs[f'rule{rule_num}'] = round(float(auc_r), 4)
        out_r = os.path.join(MODELS_DIR, f'lgbm_rule{rule_num}.txt')
        m_r.save_model(out_r)
        print(f"  rule{rule_num}: {n_rule} signals  AUC={auc_r:.4f} → {out_r}")

    metrics = {
        'auc': round(auc, 4),
        'accuracy': round(float(wr), 4),
        'precision': round(float(precision), 4),
        'n_signals': int(len(df)),
        'n_estimators': int(model.num_trees()),
        'feature_importance': {k: round(v, 2) for k, v in imp},
        'regime_aucs': regime_aucs,
        'rule_aucs': rule_aucs,
    }
    return metrics


def lgbm_only_main():
    """Phase 1B in a fresh subprocess — all HMM heap reclaimed by OS before this runs."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 60)
    print("  PHASE 1B: LightGBM (fresh subprocess)")
    print("=" * 60)

    with open(os.path.join(MODELS_DIR, 'hmm_params.json')) as f:
        params = json.load(f)
    current_regime   = params['current_regime']
    regime_map       = {int(k): v for k, v in params['regime_map'].items()}
    state_by_date    = params['state_by_date']
    with open(os.path.join(MODELS_DIR, 'hmm_posteriors.json')) as f:
        posterior_by_date = json.load(f)

    metrics_lgbm = train_lgbm(current_regime, state_by_date=state_by_date,
                               regime_map=regime_map, posterior_by_date=posterior_by_date)
    out = os.path.join(MODELS_DIR, 'phase1_metrics.json')
    with open(out, 'w') as f:
        json.dump({'lgbm': metrics_lgbm, 'regime': current_regime}, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Phase 1 complete. Current regime: {current_regime}")
    print("  Next: python scripts/ml/train_phase2.py")
    print("=" * 60)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 60)
    print("  PHASE 1: HMM + LightGBM Training")
    print("=" * 60)

    train_hmm()
    # Spawn LightGBM in a fresh subprocess so Windows OS reclaims all HMM heap pages
    import subprocess
    _this = os.path.abspath(__file__)
    _base = os.path.dirname(os.path.dirname(os.path.dirname(_this)))
    sig_csv = os.path.join(MODELS_DIR, 'signal_dataset.csv')
    if not os.path.exists(sig_csv):
        print(f"  WARN: signal_dataset.csv not found — skipping Phase 1B LGBM training")
        print(f"  HMM models saved. LGBM requires signal_dataset.csv to retrain.")
        return
    result = subprocess.run(
        [sys.executable, _this, '--lgbm-only'],
        cwd=_base, check=False
    )
    if result.returncode != 0:
        print(f"  WARN: Phase 1B LGBM failed (exit {result.returncode}) — HMM still saved")


if __name__ == '__main__':
    if '--lgbm-only' in sys.argv:
        lgbm_only_main()
    else:
        main()
