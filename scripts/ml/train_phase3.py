"""
Phase 3: LSTM Sequence Model + Ensemble Stacking Meta-Learner

Run AFTER Phase 2:
    python scripts/ml/train_phase3.py

Outputs:
    models/lstm_model.pt            — PyTorch LSTM weights
    models/stacking_weights.json    — Meta-learner logistic regression weights
    models/phase3_metrics.json      — Metrics

Prerequisites:
    Phases 1 & 2 complete.
    pip install torch
"""

import os, json, sys, warnings
import numpy as np
import pandas as pd
try:
    from tqdm import tqdm as _tqdm
    def _progress(it, **kw): return _tqdm(it, **kw)
except ImportError:
    def _progress(it, total=None, desc='', **kw):
        n = total or 0
        for i, x in enumerate(it):
            if i % 10000 == 0:
                print(f"  {desc}: {i}/{n}", flush=True)
            yield x

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.ml.data_utils import (
    DATA_FILE, MODELS_DIR, FEATURE_COLS, SEQUENCE_COLS, WIN_COL, MONOTONE_CONSTRAINTS,
    load_signal_dataset, ema, rsi_wilder, sg_vel as sg_velocity
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from scripts.ml.lstm_model import LSTMSignalModel, HIDDEN_DIM, DROPOUT
from scripts.ml.cv_utils import PurgedTimeSeriesSplit

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN    = 30
EPOCHS     = 40
BATCH_SIZE = 128
LR         = 1e-3
PATIENCE   = 8

# LSTM AUC=0.5882 (near-random) drags ensemble down — disable until Sprint 4 rebuild
USE_LSTM   = False


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels,    dtype=torch.float32)

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ─────────────────────── Feature sequence builder ────────────────────────────

def _precompute_features(grp):
    """Vectorised per-symbol feature matrix. Returns DataFrame indexed by Date."""
    c = grp['Close'].values.astype(np.float64)
    h = grp['High'].values.astype(np.float64)
    l = grp['Low'].values.astype(np.float64)
    v = grp['Volume'].values.astype(np.float64) if 'Volume' in grp.columns else np.ones(len(c))
    n = len(c)

    ret   = np.zeros(n, np.float32)
    hl_r  = np.zeros(n, np.float32)
    vol_r = np.ones(n,  np.float32)
    rsi14 = np.full(n, 50.0, np.float32)
    sgv   = np.zeros(n, np.float32)
    mom5  = np.zeros(n, np.float32)

    for j in range(1, n):
        if c[j-1] > 0:
            ret[j] = float(c[j] / c[j-1] - 1)
        if c[j] > 0:
            hl_r[j] = float((h[j] - l[j]) / c[j])
        vol20 = float(v[max(0, j-20):j].mean()) if j > 0 else 1.0
        vol_r[j] = float(min(v[j] / vol20, 5.0)) if vol20 > 0 else 1.0
        rsi14[j] = float(rsi_wilder(c[max(0, j-29):j+1]))
        sgv[j]   = float(sg_velocity(c[max(0, j-20):j+1]))
        if j >= 5 and c[j-5] > 0:
            mom5[j] = float(c[j] / c[j-5] - 1)

    feat = np.stack([ret, hl_r, vol_r, rsi14 / 100.0, sgv, mom5], axis=1).astype(np.float32)
    return feat  # shape [n, 6], aligned with grp.index


def build_sequences(df_all, df_signals):
    """
    Pre-compute all features per symbol once, then slice per signal.
    ~5-10x faster than computing features inside the signal loop.
    """
    df_all = df_all.sort_values('Date')

    # Pre-compute feature matrix for every symbol
    print("  Pre-computing features per symbol...")
    sym_feat  = {}   # sym -> (feat_array [n,6], date_index [n])
    for sym, grp in _progress(df_all.groupby('Symbol'),
                               total=df_all['Symbol'].nunique(),
                               desc='symbols', leave=False):
        feat = _precompute_features(grp.reset_index(drop=True))
        sym_feat[sym] = (feat, pd.to_datetime(grp['Date']).values.astype('datetime64[ns]'))

    print(f"  Features ready for {len(sym_feat)} symbols. Building sequences...")

    sequences, labels, valid_pos = [], [], []
    skipped = 0
    n_sig   = len(df_signals)

    for pos_i, (_, sig) in enumerate(_progress(df_signals.iterrows(), total=n_sig,
                                                desc='sequences', leave=True)):
        sym  = sig['symbol']
        date = np.datetime64(pd.Timestamp(sig['date']), 'ns')
        entry = sym_feat.get(sym)
        if entry is None:
            skipped += 1
            continue

        feat_arr, dates = entry
        # Find position of last bar <= signal date
        bar_pos = np.searchsorted(dates, date, side='right') - 1
        if bar_pos < SEQ_LEN + 4:
            skipped += 1
            continue

        seq = feat_arr[bar_pos - SEQ_LEN + 1: bar_pos + 1]   # shape [SEQ_LEN, 6]
        if len(seq) < SEQ_LEN:
            skipped += 1
            continue

        sequences.append(seq)
        labels.append(int(sig[WIN_COL]))
        valid_pos.append(pos_i)

    print(f"  Built {len(sequences)} sequences, skipped {skipped}")
    return (np.array(sequences, dtype=np.float32),
            np.array(labels, dtype=np.int32),
            valid_pos)


# ──────────────────────────── LSTM Training ───────────────────────────────────

def train_lstm(X_seq_tr, y_tr, X_seq_te, y_te):
    model   = LSTMSignalModel(input_dim=X_seq_tr.shape[2]).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    pos_w   = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                            dtype=torch.float32).to(DEVICE)

    tr_ds = SequenceDataset(X_seq_tr, y_tr)
    te_ds = SequenceDataset(X_seq_te, y_te)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_auc, best_state, patience_cnt = 0.0, None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for Xb, yb in tr_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            # Weight positive samples by pos_weight
            pred  = model(Xb)
            wt    = torch.where(yb == 1, pos_w.squeeze(), torch.ones(1).to(DEVICE))
            loss  = (wt * nn.functional.binary_cross_entropy(pred, yb, reduction='none')).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for Xb, yb in te_dl:
                all_p.append(model(Xb.to(DEVICE)).cpu().numpy())
                all_y.append(yb.numpy())
        p = np.concatenate(all_p)
        yt = np.concatenate(all_y)
        auc = roc_auc_score(yt, p) if len(np.unique(yt)) > 1 else 0.5

        if epoch % 5 == 0:
            print(f"    Epoch {epoch:3d}  AUC={auc:.4f}")

        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"    Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    print(f"  LSTM best AUC: {best_auc:.4f}")
    return model, best_auc


def get_lstm_probs(model, X_seq):
    model.eval()
    with torch.no_grad():
        ds  = DataLoader(SequenceDataset(X_seq, np.zeros(len(X_seq))),
                         batch_size=512, shuffle=False, num_workers=0)
        return np.concatenate([model(Xb.to(DEVICE)).cpu().numpy() for Xb, _ in ds])


# ────────────────────────── Focal Loss (XGBoost) ─────────────────────────────

def focal_loss_obj(y_pred, dtrain, gamma=2.0, alpha=0.25):
    """
    Focal loss custom objective for xgb.train().
    Downweights easy (well-classified) samples — better for imbalanced labels
    than scale_pos_weight alone.
    Returns (gradient, hessian) per sample.
    """
    y = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -20, 20)))
    pt = np.where(y == 1, p, 1 - p)          # probability of true class
    at = np.where(y == 1, alpha, 1 - alpha)   # class-balanced alpha
    focal_wt = at * (1 - pt) ** gamma
    grad = focal_wt * (p - y)
    hess = focal_wt * p * (1 - p)
    return grad, hess




# ─────────────────────── Stacking Meta-Learner ───────────────────────────────

def _load_regime_lgbm_models():
    """Load per-regime LightGBM sub-models. Prefers Phase 2c models, falls back to Phase 1."""
    models = {}
    for state in range(4):
        # Phase 2c regime-conditional models take priority
        p2c = os.path.join(MODELS_DIR, f'lgbm2c_regime_{state}.txt')
        p1  = os.path.join(MODELS_DIR, f'lgbm_regime_{state}.txt')
        path = p2c if os.path.exists(p2c) else (p1 if os.path.exists(p1) else None)
        if path:
            models[state] = lgb.Booster(model_file=path)
            print(f"    Regime {state}: loaded {os.path.basename(path)}")
    return models


def _regime_lgbm_probs(X_tab, dates, state_by_date, regime_models, global_lgbm):
    """Route each signal to its HMM-state model; fall back to global. Batched."""
    states_arr = np.array([state_by_date.get(str(d)[:10], -1) for d in dates])
    probs = np.zeros(len(X_tab), dtype=np.float32)
    for s in range(4):
        mask = states_arr == s
        if not mask.any():
            continue
        m = regime_models.get(s, global_lgbm)
        probs[mask] = m.predict(X_tab[mask]).astype(np.float32)
    # signals with no known state fall back to global model
    fallback_mask = ~np.isin(states_arr, list(regime_models.keys()))
    if fallback_mask.any():
        probs[fallback_mask] = global_lgbm.predict(X_tab[fallback_mask]).astype(np.float32)
    return probs


def _soft_blend_probs(X_tab, dates, posterior_by_date, regime_models, global_lgbm):
    """Soft-blend: weight each sub-model by HMM posterior P(state|obs)."""
    if not regime_models or not posterior_by_date:
        return global_lgbm.predict(X_tab)
    # Pre-compute all sub-model predictions at once (vectorised)
    preds = np.stack(
        [regime_models.get(s, global_lgbm).predict(X_tab) for s in range(4)],
        axis=1,
    )  # [N, 4]
    posts = np.array(
        [posterior_by_date.get(str(d)[:10], [0.25, 0.25, 0.25, 0.25]) for d in dates],
        dtype=np.float32,
    )  # [N, 4]
    return (posts * preds).sum(axis=1)


def train_stacking(lgbm_model, lstm_model,
                   X_tab, X_seq, y, df_sig=None, use_lstm=True):
    """
    Generate OOS predictions from each base model then fit a logistic meta-learner.
    Uses k=5 fold cross-predictions to avoid leakage.
    df_sig: signal DataFrame (needed for regime-aware LightGBM routing).
    """
    # Load regime sub-models, date→state mapping, and HMM posteriors (Phase 1 outputs)
    regime_models = _load_regime_lgbm_models()
    state_by_date  = {}
    posterior_by_date = {}
    hmm_path = os.path.join(MODELS_DIR, 'hmm_params.json')
    if os.path.exists(hmm_path):
        with open(hmm_path) as f:
            hmm_data = json.load(f)
        state_by_date = hmm_data.get('state_by_date', {})
    post_path = os.path.join(MODELS_DIR, 'hmm_posteriors.json')
    if os.path.exists(post_path):
        with open(post_path) as f:
            posterior_by_date = json.load(f)
    use_regime = bool(regime_models and state_by_date and df_sig is not None)
    use_soft   = bool(regime_models and posterior_by_date and df_sig is not None)
    if use_regime:
        print(f"  Regime-aware stacking: {len(regime_models)} sub-models loaded.")
        print(f"  Soft blending: {'enabled' if use_soft else 'disabled (no posteriors)'}.")
    else:
        print("  Regime sub-models not found — using global LightGBM only.")

    dates = df_sig['date'].values if df_sig is not None else np.array([''] * len(y))

    n        = len(y)
    xgb_oos  = np.zeros(n)
    lgb_oos  = np.zeros(n)
    lst_oos  = np.zeros(n)
    reg_oos  = np.zeros(n)   # hard-routed regime LightGBM
    soft_oos = np.zeros(n)   # soft-blended regime LightGBM

    splitter = PurgedTimeSeriesSplit(n_splits=5, embargo_days=7)
    signal_dates = dates if df_sig is not None else None

    for fold, (tr_idx, val_idx) in enumerate(splitter.split(X_tab, dates=signal_dates)):
        print(f"    Stacking fold {fold+1}/5  (train={len(tr_idx)}  val={len(val_idx)})")
        Xtr_t, Xval_t = X_tab[tr_idx], X_tab[val_idx]
        Xtr_s, Xval_s = X_seq[tr_idx], X_seq[val_idx]
        ytr, yval     = y[tr_idx], y[val_idx]

        # XGBoost OOS — focal loss + monotone constraints
        dtrain = xgb.DMatrix(Xtr_t.astype(np.float32), label=ytr)
        dval   = xgb.DMatrix(Xval_t.astype(np.float32))
        m_xgb  = xgb.train(
            {'max_depth': 5, 'eta': 0.1, 'seed': 42,
             'monotone_constraints': tuple(MONOTONE_CONSTRAINTS)},
            dtrain, num_boost_round=200,
            obj=focal_loss_obj,
        )
        # Custom obj returns raw scores → apply sigmoid for probabilities
        xgb_oos[val_idx] = 1.0 / (1.0 + np.exp(-m_xgb.predict(dval)))

        # LightGBM OOS (global)
        dtrl = lgb.Dataset(Xtr_t, label=ytr)
        m_lgb = lgb.train({'objective': 'binary', 'verbose': -1, 'seed': 42,
                           'num_leaves': 31, 'learning_rate': 0.05},
                          dtrl, num_boost_round=200)
        lgb_oos[val_idx] = m_lgb.predict(Xval_t)

        # LSTM OOS (skipped when USE_LSTM=False — AUC was 0.5882, near-random)
        if use_lstm and lstm_model is not None:
            lst_oos[val_idx] = get_lstm_probs(lstm_model, Xval_s)

        # Regime-aware LightGBM — hard routing (argmax state)
        if use_regime:
            reg_oos[val_idx] = _regime_lgbm_probs(
                Xval_t, dates[val_idx], state_by_date, regime_models, lgbm_model)
        else:
            reg_oos[val_idx] = lgb_oos[val_idx]

        # Soft-blend routing (posterior-weighted)
        if use_soft:
            soft_oos[val_idx] = _soft_blend_probs(
                Xval_t, dates[val_idx], posterior_by_date, regime_models, lgbm_model)
        else:
            soft_oos[val_idx] = reg_oos[val_idx]

    # ── Stacking XGB Optuna: tune on fold-1 data, apply to all folds ────────
    # (already ran fold-1 XGB above; re-run a quick search to find best params)
    print("\n  Tuning stacking XGB params (15 trials on fold-1 data)…")
    fold1_tr, fold1_val = next(iter(
        PurgedTimeSeriesSplit(n_splits=5, embargo_days=7).split(X_tab, dates=signal_dates)
    ))
    dtrain_tune = xgb.DMatrix(X_tab[fold1_tr].astype(np.float32), label=y[fold1_tr])
    dval_tune   = xgb.DMatrix(X_tab[fold1_val].astype(np.float32), label=y[fold1_val])

    def _xgb_obj_tune(trial):
        p = {
            'max_depth':        trial.suggest_int('max_depth', 3, 7),
            'eta':              trial.suggest_float('eta', 0.03, 0.2, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'seed': 42,
            'monotone_constraints': tuple(MONOTONE_CONSTRAINTS),
            # custom obj outputs raw logits; disable built-in eval metric
            # (early stopping via Optuna trial pruning instead of xgb ES)
        }
        m = xgb.train(p, dtrain_tune, num_boost_round=200, obj=focal_loss_obj)
        raw = m.predict(dval_tune)
        prob = 1.0 / (1.0 + np.exp(-raw))
        return roc_auc_score(y[fold1_val], prob)

    try:
        import optuna as _optuna
        _optuna.logging.set_verbosity(_optuna.logging.WARNING)
        xgb_study = _optuna.create_study(direction='maximize',
                                         sampler=_optuna.samplers.TPESampler(seed=42))
        xgb_study.optimize(_xgb_obj_tune, n_trials=15, show_progress_bar=False)
        best_xgb_params = {**xgb_study.best_params,
                           'seed': 42,
                           'monotone_constraints': tuple(MONOTONE_CONSTRAINTS)}
        print(f"  Best stacking XGB AUC={xgb_study.best_value:.4f}  params={xgb_study.best_params}")
    except Exception as e:
        print(f"  XGB Optuna skipped ({e}) — using default params.")
        best_xgb_params = {'max_depth': 5, 'eta': 0.1, 'seed': 42,
                           'monotone_constraints': tuple(MONOTONE_CONSTRAINTS)}

    # Re-run all 5 folds using tuned XGB params (overwrite xgb_oos)
    xgb_oos_tuned = np.zeros(n)
    for fold, (tr_idx, val_idx) in enumerate(splitter.split(X_tab, dates=signal_dates)):
        dtrain_f = xgb.DMatrix(X_tab[tr_idx].astype(np.float32), label=y[tr_idx])
        dval_f   = xgb.DMatrix(X_tab[val_idx].astype(np.float32))
        m_f = xgb.train(best_xgb_params, dtrain_f, num_boost_round=200, obj=focal_loss_obj)
        raw = m_f.predict(dval_f)
        xgb_oos_tuned[val_idx] = 1.0 / (1.0 + np.exp(-raw))
    xgb_oos = xgb_oos_tuned   # replace with tuned version

    # ── Meta-learner: shallow LightGBM (non-linear combination of base models) ─
    print("  Fitting LightGBM meta-learner…")
    base_cols = [xgb_oos, lgb_oos, reg_oos, soft_oos]
    if use_lstm and lstm_model is not None:
        base_cols.insert(2, lst_oos)   # position preserved for backward compat
    meta_X = np.column_stack(base_cols)

    # Temporal split for meta-learner (last 20% as val — avoid leakage)
    n_meta_tr = int(n * 0.80)
    meta_lgb_tr = lgb.Dataset(meta_X[:n_meta_tr], label=y[:n_meta_tr])
    meta_lgb_val = lgb.Dataset(meta_X[n_meta_tr:], label=y[n_meta_tr:],
                               reference=meta_lgb_tr)
    meta_params = {
        'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'seed': 42,
        'num_leaves': 8, 'max_depth': 3, 'learning_rate': 0.05,
        'feature_fraction': 1.0, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'min_child_samples': 50,   # prevents overfit on 5 meta-features
    }
    meta_lgb = lgb.train(meta_params, meta_lgb_tr, num_boost_round=300,
                         valid_sets=[meta_lgb_val],
                         callbacks=[lgb.early_stopping(30, verbose=False)])
    meta_lgb_path = os.path.join(MODELS_DIR, 'meta_lgbm.txt')
    meta_lgb.save_model(meta_lgb_path)
    print(f"  Meta-LightGBM saved → {meta_lgb_path}")

    # OOS eval on held-out slice only (avoids in-sample AUC inflation)
    final_preds_oos = meta_lgb.predict(meta_X[n_meta_tr:])
    auc_meta        = roc_auc_score(y[n_meta_tr:], final_preds_oos)
    # Full-set preds needed for base-model AUC reporting only
    final_preds = meta_lgb.predict(meta_X)
    auc_xgb     = roc_auc_score(y, xgb_oos)
    auc_lgb     = roc_auc_score(y, lgb_oos)
    auc_lst     = roc_auc_score(y, lst_oos) if (use_lstm and lstm_model is not None) else None
    auc_reg     = roc_auc_score(y, reg_oos)
    auc_soft    = roc_auc_score(y, soft_oos)

    print(f"\n  Stacking OOS AUC:")
    print(f"    XGBoost (tuned):  {auc_xgb:.4f}")
    print(f"    LightGBM:         {auc_lgb:.4f}")
    if auc_lst is not None:
        print(f"    LSTM:             {auc_lst:.4f}")
    else:
        print(f"    LSTM:             disabled (USE_LSTM=False)")
    print(f"    Regime-Hard:      {auc_reg:.4f}")
    print(f"    Regime-Soft:      {auc_soft:.4f}")
    print(f"    STACK (LGBM meta):{auc_meta:.4f}  ← ensemble")

    # ── Threshold calibration on held-out set only (avoids in-sample leakage) ──
    print("\n  Calibrating decision threshold…")
    y_cal_t = y[n_meta_tr:]
    thresholds = np.arange(0.30, 0.80, 0.01)
    best_thresh, best_prec = 0.5, 0.0
    for t in thresholds:
        preds_bin = (final_preds_oos >= t).astype(int)
        tp = (preds_bin & y_cal_t).sum()
        fp = (preds_bin & (1 - y_cal_t)).sum()
        fn = ((1 - preds_bin) & y_cal_t).sum()
        prec   = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if recall >= 0.30 and prec > best_prec:
            best_prec, best_thresh = prec, t
    print(f"  Optimal threshold: {best_thresh:.2f}  "
          f"(precision={best_prec:.3f} at recall≥0.30)")

    weights = {
        'meta_learner':  'lgbm',
        'meta_lgbm_path': meta_lgb_path,
        'use_lstm':      use_lstm and lstm_model is not None,
        'auc_xgb':       round(auc_xgb, 4),
        'auc_lgbm':      round(auc_lgb, 4),
        'auc_lstm':      round(auc_lst, 4) if auc_lst is not None else None,
        'auc_regime':    round(auc_reg, 4),
        'auc_soft':      round(auc_soft, 4),
        'auc_stack':     round(auc_meta, 4),
        'threshold':     round(float(best_thresh), 2),
        'threshold_precision': round(float(best_prec), 4),
        'best_xgb_params': {k: v for k, v in best_xgb_params.items()
                            if k != 'monotone_constraints'},
    }
    return weights


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 60)
    print("  PHASE 3: LSTM + Stacking Meta-Learner")
    print("=" * 60)
    print(f"  Device: {DEVICE}")

    # Prefer Phase 2c global model; fall back to Phase 1
    _p2c_global = os.path.join(MODELS_DIR, 'lgbm2c_global.txt')
    _p1_global  = os.path.join(MODELS_DIR, 'lgbm_model.txt')
    _lgbm_path  = _p2c_global if os.path.exists(_p2c_global) else _p1_global
    print(f"  Loading global LightGBM: {os.path.basename(_lgbm_path)}")
    lgbm_model = lgb.Booster(model_file=_lgbm_path)

    # Load signal dataset
    df_sig = load_signal_dataset()

    print(f"\n  {len(df_sig)} signals loaded.")

    # Cap at 100k for LSTM — halves peak RAM vs 200k; still robust for sequence training
    if len(df_sig) > 100_000:
        df_sig = df_sig.sample(100_000, random_state=42).sort_values('date').reset_index(drop=True)
        print(f"  Sampled 100k signals for LSTM sequence building.")

    X_tab  = df_sig[FEATURE_COLS].values.astype(np.float32)
    y      = df_sig[WIN_COL].values.astype(int)

    # Build LSTM sequences from raw OHLCV
    print("\n── Phase 3A: Building 30-bar sequences ─────────────────────────")
    data_file = DATA_FILE
    if not os.path.exists(data_file):
        print(f"  WARNING: {data_file} not found. Using tabular features as sequence proxy.")
        # Fallback: repeat tabular features as dummy sequence
        X_seq = np.tile(X_tab[:, :len(SEQUENCE_COLS), np.newaxis].transpose(0,2,1),
                        (1, SEQ_LEN, 1))
    else:
        df_all = pd.read_csv(data_file)
        df_all.columns = [c.strip().title() for c in df_all.columns]
        df_all['Date'] = pd.to_datetime(df_all['Date'], dayfirst=True)
        X_seq, y_seq, seq_valid_pos = build_sequences(df_all, df_sig)
        print(f"  Built {len(X_seq)} sequences out of {len(df_sig)} signals.")
        # Re-align tabular data to exactly the signals that produced sequences
        df_sig = df_sig.iloc[seq_valid_pos].reset_index(drop=True)
        X_tab  = df_sig[FEATURE_COLS].values.astype(np.float32)
        y      = y_seq

    # Temporal split
    n_tr  = int(len(y) * 0.80)
    X_t_tr, X_t_te = X_tab[:n_tr], X_tab[n_tr:]
    X_s_tr, X_s_te = X_seq[:n_tr], X_seq[n_tr:]
    y_tr,   y_te   = y[:n_tr],     y[n_tr:]

    # Phase 3A: Train LSTM (disabled — AUC was 0.5882, near-random; Sprint 4 will rebuild with OHLCV sequences)
    print("\n── Phase 3A: LSTM ───────────────────────────────────────────────")
    if USE_LSTM:
        lstm_model, lstm_auc = train_lstm(X_s_tr, y_tr, X_s_te, y_te)
        out_lstm = os.path.join(MODELS_DIR, 'lstm_model.pt')
        torch.save({
            'model_state': lstm_model.state_dict(),
            'input_dim':   len(SEQUENCE_COLS),
            'hidden_dim':  HIDDEN_DIM,
            'seq_len':     SEQ_LEN,
            'seq_cols':    SEQUENCE_COLS,
        }, out_lstm)
        print(f"  Saved → {out_lstm}")
    else:
        lstm_model, lstm_auc = None, 0.0
        print("  LSTM disabled (USE_LSTM=False). Skipping — stack will use XGB+LGB+Regime+Soft.")

    # Phase 3B: Stacking
    print("\n── Phase 3B: Stacking Meta-Learner ─────────────────────────────")
    stacking_weights = train_stacking(
        lgbm_model, lstm_model,
        X_tab, X_seq, y, df_sig=df_sig, use_lstm=USE_LSTM
    )

    out_sw = os.path.join(MODELS_DIR, 'stacking_weights.json')
    with open(out_sw, 'w') as f:
        json.dump(stacking_weights, f, indent=2)
    print(f"  Saved → {out_sw}")

    with open(os.path.join(MODELS_DIR, 'phase3_metrics.json'), 'w') as f:
        json.dump({'lstm_auc': round(lstm_auc, 4) if lstm_auc else None, **stacking_weights}, f, indent=2)

    print("\n" + "=" * 60)
    print("  Phase 3 complete.")
    print("  Next: python scripts/ml/train_phase4.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
