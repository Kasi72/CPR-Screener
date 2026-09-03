"""
cpr_phase3_kernel.py — Self-contained Phase 3 (LSTM + Stacking) for Kaggle GPU.

Reads from /kaggle/input/cpr-screener-phase3-inputs/:
  signal_dataset.csv
  models/lgbm_model.txt
  models/lgbm_regime_0..3.txt        (optional, Phase 1 regime sub-models)
  models/lgbm_rule1..11.txt          (optional, Phase 1 rule sub-models)
  models/hmm_params.json             (optional, HMM state-by-date mapping)
  models/hmm_posteriors.json         (optional, soft-blend posteriors)
  ohlcv/ALL_SYMBOLS_OHLCV.csv        (optional, enables real 30-bar sequences)

Writes to /kaggle/working/:
  lstm_model.pt
  stacking_weights.json
  meta_lgbm.txt
  phase3_metrics.json

KEEP IN SYNC with: data_utils.py (FEATURE_COLS, MONOTONE_CONSTRAINTS, WIN_COL)
                   lstm_model.py  (LSTMSignalModel, HIDDEN_DIM, DROPOUT)
                   cv_utils.py    (PurgedTimeSeriesSplit)
                   train_phase3.py (training logic)
"""

import os, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')

INPUT  = '/kaggle/input/cpr-screener-phase3-inputs'
MODELS = os.path.join(INPUT, 'models')
WORK   = '/kaggle/working'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Verify CUDA LSTM actually runs on this Kaggle instance.
# cudaErrorNoKernelImageForDevice = PyTorch binary not compiled for this GPU's
# compute capability (sm mismatch). Fall back to CPU rather than crash.
if DEVICE.type == 'cuda':
    try:
        _test = torch.nn.LSTM(4, 8, batch_first=True).to(DEVICE)
        _inp  = torch.zeros(2, 5, 4, device=DEVICE)
        _test(_inp)
        del _test, _inp
        print(f"Device: {DEVICE}  (CUDA LSTM verified ✓)")
    except Exception as _e:
        print(f"CUDA LSTM probe failed ({type(_e).__name__}: {_e})")
        print("Falling back to CPU — will be slower but will complete.")
        DEVICE = torch.device('cpu')
else:
    print(f"Device: {DEVICE}")

os.makedirs(WORK, exist_ok=True)


# ── Constants (must match data_utils.py) ──────────────────────────────────────

_BASE_FEATURES = [
    'cpr_width_pct', 'vwap_dist', 'atr_pct_rank', 'vol_rank',
    'n_rules_fired', 'sg_vel', 'ema200_dist', 'rsi14',
    'mom5', 'dow', 'rule_id', 'direction',
    'dist_hi52', 'dist_lo52', 'vol_accel',
    'market_rs_5d', 'market_rs_20d', 'sector_rs_5d', 'sector_rs_20d',
    'deliv_pct', 'pcr', 'india_vix', 'conf_vol', 'rsi_dir', 'hi52_dir',
    'cpr_compress', 'cpr_pos', 'dist_r1', 'dist_s1',
    'mom3', 'mom10', 'mom20', 'rsi_div', 'vol_accel_delta',
    'days_since_52hi', 'expiry_dist',
    # Sprint 1 CPR features (Phase 2c)
    'cpr_overlap_pct', 'open_to_cpr_dist', 'prev_cpr_respected', 'cpr_zone_vol_ratio',
    # HMM regime
    'hmm_regime',
    # Sprint 2A: compression/structure CPR features
    'open_inside_cpr', 'cpr_virgin', 'consecutive_narrow_cprs',
    'cpr_midpoint_trend', 'cpr_expansion_factor',
    # Sprint 2B: structural + context CPR features
    'cpr_above_prev_cpr', 'prev_close_inside_cpr', 'atr_to_cpr_ratio',
    'cpr_width_percentile_252d', 'prev_day_ochoa_type',
]  # 46

_INTERACTION_FEATURES = [
    'cpr_vol_interaction',       # cpr_compress x vol_rank
    'regime_momentum',           # hmm_regime x mom5
    'cpr_rsi_squeeze',           # (1 - cpr_width_pct) x rsi14
    'overlap_vol_signal',        # cpr_overlap_pct x cpr_zone_vol_ratio
    'rs_direction_alignment',    # (market_rs_5d + sector_rs_5d) x direction
    'virgin_momentum',           # cpr_virgin x mom5
    'narrow_breakout_vol',       # consecutive_narrow_cprs x vol_rank
]  # 7

FEATURE_COLS = _BASE_FEATURES + _INTERACTION_FEATURES  # 58 — matches Phase 2c model (51 base + 7 interactions)

# Directional features (sign flipped for SELL signals)
_DIRECTIONAL = {
    'dist_hi52', 'dist_lo52', 'vwap_dist', 'ema200_dist',
    'mom3', 'mom5', 'mom10', 'mom20',
    'market_rs_5d', 'market_rs_20d', 'sector_rs_5d', 'sector_rs_20d',
    'cpr_pos', 'dist_r1', 'dist_s1', 'sg_vel', 'open_to_cpr_dist',
}

MONOTONE_CONSTRAINTS = [
    0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 1, 0, -1, 1, -1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -1,
]

SEQUENCE_COLS = ['ret', 'hl_range', 'vol_ratio', 'rsi14', 'sg_vel', 'mom5',
                 'ema14_dist', 'atr14', 'bb_pos', 'vol_mom']
WIN_COL       = 'hit_t1'
SEQ_LEN       = 30
EPOCHS        = 80
BATCH_SIZE    = 512   # larger batch on GPU (T4 = 16 GB)
LR            = 5e-4
PATIENCE      = 15
HIDDEN_DIM    = 128
DROPOUT       = 0.3


# ── LSTM model (mirrors lstm_model.py) ────────────────────────────────────────

class LSTMSignalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=2, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attn = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        out, _  = self.lstm(x)
        attn_w  = torch.softmax(self.attn(out), dim=1)
        ctx     = (attn_w * out).sum(dim=1)
        return self.head(ctx).squeeze(-1)


# ── CV util (mirrors cv_utils.py) ─────────────────────────────────────────────

class PurgedTimeSeriesSplit:
    def __init__(self, n_splits=5, embargo_days=7):
        self.n_splits     = n_splits
        self.embargo_days = embargo_days

    def split(self, X, y=None, dates=None):
        n         = len(X)
        fold_size = n // (self.n_splits + 1)
        for fold in range(self.n_splits):
            val_start = fold_size * (fold + 1)
            val_end   = min(val_start + fold_size, n)
            if dates is not None:
                val_date    = pd.Timestamp(str(dates[val_start])[:10])
                cutoff      = val_date - pd.Timedelta(days=self.embargo_days)
                train_dates = pd.to_datetime([str(d)[:10] for d in dates[:val_start]])
                train_end   = int((train_dates <= cutoff).sum())
            else:
                train_end = max(0, val_start - 20)
            tr_idx  = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            if len(tr_idx) >= 100 and len(val_idx) >= 50:
                yield tr_idx, val_idx


# ── TA helpers (subset from data_utils.py) ────────────────────────────────────

def rsi_wilder(closes, period=14):
    c = np.array(closes, dtype=float)
    if len(c) < period + 2:
        return 50.0
    d = np.diff(c)
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    ag, al = g[:period].mean(), l[:period].mean()
    for j in range(period, len(d)):
        ag = (ag * (period - 1) + g[j]) / period
        al = (al * (period - 1) + l[j]) / period
    return 100.0 - (100.0 / (1 + ag / al)) if al > 0 else 100.0


def sg_vel(closes, window=11, poly=3):
    if len(closes) < window:
        return 0.0
    return float(savgol_filter(closes, window, poly, deriv=1)[-1])


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_signal_dataset():
    path = os.path.join(INPUT, 'signal_dataset.csv')
    print(f"Loading {path} ...")
    _num = {c: 'float32' for c in FEATURE_COLS if c != 'rule_id'}
    _num.update({'hit_t1': 'float32', 'win': 'float32',
                 'win_rr': 'float32', 'rr_ratio': 'float32'})
    try:
        df = pd.read_csv(path, dtype=_num, memory_map=True)
    except Exception:
        df = pd.read_csv(path, memory_map=True, low_memory=False)
    if 'rule_id' in df.columns and df['rule_id'].dtype == object:
        df['rule_id'] = df['rule_id'].str.replace('rule', '', regex=False).astype(int)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    if WIN_COL not in df.columns:
        raise ValueError(f"Target column '{WIN_COL}' missing from dataset.")
    df[WIN_COL] = df[WIN_COL].astype(int)

    # Sprint 4: load Phase 2b/2c meta-scores if present (neutral 0.5 if missing)
    for score_col in ('lgbm2b_score', 'lgbm2c_score'):
        if score_col not in df.columns:
            df[score_col] = 0.5
        else:
            df[score_col] = df[score_col].fillna(0.5).clip(0.0, 1.0).astype('float32')

    p2b_real = (df['lgbm2b_score'] != 0.5).mean()
    p2c_real = (df['lgbm2c_score'] != 0.5).mean()
    print(f"  {len(df)} signals loaded.  "
          f"lgbm2b_score coverage={p2b_real:.1%}  lgbm2c_score coverage={p2c_real:.1%}")
    return df


# ── Sequence builder ──────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels,    dtype=torch.float32)

    def __len__(self):             return len(self.y)
    def __getitem__(self, i):      return self.X[i], self.y[i]


def _precompute_features(grp):
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
    ema14     = np.zeros(n, np.float32)
    ema14_d   = np.zeros(n, np.float32)
    atr14     = np.zeros(n, np.float32)
    atr14_s   = np.zeros(n, np.float32)
    bb_pos    = np.zeros(n, np.float32)
    vol_mom   = np.zeros(n, np.float32)

    _ema_alpha = 2.0 / 15.0
    ema14[0]  = c[0]
    atr14_s[0] = (h[0] - l[0]) / c[0] if c[0] > 0 else 0.0

    for j in range(1, n):
        if c[j-1] > 0: ret[j]  = float(c[j] / c[j-1] - 1)
        if c[j]   > 0: hl_r[j] = float((h[j] - l[j]) / c[j])
        vol20 = float(v[max(0, j-20):j].mean()) if j > 0 else 1.0
        vol_r[j] = float(min(v[j] / vol20, 5.0)) if vol20 > 0 else 1.0
        rsi14[j] = float(rsi_wilder(c[max(0, j-29):j+1]))
        sgv[j]   = float(sg_vel(c[max(0, j-20):j+1]))
        if j >= 5 and c[j-5] > 0: mom5[j] = float(c[j] / c[j-5] - 1)
        # EMA14
        ema14[j] = _ema_alpha * c[j] + (1 - _ema_alpha) * ema14[j-1]
        if ema14[j] > 0: ema14_d[j] = float((c[j] - ema14[j]) / ema14[j])
        # ATR14 (Wilder smoothing)
        tr = float(max(h[j] - l[j], abs(h[j] - c[j-1]), abs(l[j] - c[j-1])))
        raw_atr = tr / c[j] if c[j] > 0 else 0.0
        atr14_s[j] = (atr14_s[j-1] * 13 + raw_atr) / 14
        # Bollinger Band position (20-day)
        if j >= 20:
            w = c[j-20:j]; mu = w.mean(); sd = w.std()
            if sd > 0: bb_pos[j] = float(np.clip((c[j] - mu) / (2 * sd), -2, 2))
        # Volume momentum
        if j >= 5: vol_mom[j] = float(np.clip(vol_r[j] - vol_r[j-5], -3, 3))

    return np.stack([ret, hl_r, vol_r, rsi14 / 100.0, sgv, mom5,
                     ema14_d, atr14_s, bb_pos, vol_mom], axis=1).astype(np.float32)


def build_sequences(df_all, df_signals):
    df_all = df_all.sort_values('Date')
    print("  Pre-computing features per symbol ...")
    sym_feat = {}
    for sym, grp in df_all.groupby('Symbol'):
        feat = _precompute_features(grp.reset_index(drop=True))
        sym_feat[sym] = (feat, pd.to_datetime(grp['Date']).values.astype('datetime64[ns]'))
    print(f"  Features ready for {len(sym_feat)} symbols. Building sequences ...")
    sequences, labels, valid_pos = [], [], []
    skipped = 0
    for pos_i, (_, sig) in enumerate(df_signals.iterrows()):
        if pos_i % 50000 == 0:
            print(f"    {pos_i}/{len(df_signals)}", flush=True)
        sym   = sig['symbol']
        date  = np.datetime64(pd.Timestamp(sig['date']), 'ns')
        entry = sym_feat.get(sym)
        if entry is None: skipped += 1; continue
        feat_arr, dates = entry
        bar_pos = np.searchsorted(dates, date, side='right') - 1
        if bar_pos < SEQ_LEN + 4: skipped += 1; continue
        seq = feat_arr[bar_pos - SEQ_LEN + 1: bar_pos + 1]
        if len(seq) < SEQ_LEN: skipped += 1; continue
        sequences.append(seq)
        labels.append(int(sig[WIN_COL]))
        valid_pos.append(pos_i)
    print(f"  Built {len(sequences)} sequences, skipped {skipped}")
    return (np.array(sequences, dtype=np.float32),
            np.array(labels, dtype=np.int32), valid_pos)


# ── LSTM training ─────────────────────────────────────────────────────────────

def train_lstm(X_seq_tr, y_tr, X_seq_te, y_te):
    model   = LSTMSignalModel(input_dim=X_seq_tr.shape[2]).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    pos_w   = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                            dtype=torch.float32).to(DEVICE)
    tr_dl = DataLoader(SequenceDataset(X_seq_tr, y_tr),
                       batch_size=BATCH_SIZE, shuffle=True,  num_workers=2,
                       pin_memory=DEVICE.type == 'cuda')
    te_dl = DataLoader(SequenceDataset(X_seq_te, y_te),
                       batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                       pin_memory=DEVICE.type == 'cuda')

    best_auc, best_state, patience_cnt = 0.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for Xb, yb in tr_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(Xb)
            wt   = torch.where(yb == 1, pos_w.squeeze(), torch.ones(1).to(DEVICE))
            loss = (wt * nn.functional.binary_cross_entropy(pred, yb, reduction='none')).mean()
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
            print(f"    Epoch {epoch:3d}  AUC={auc:.4f}", flush=True)
        if auc > best_auc:
            best_auc, patience_cnt = auc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
        dl = DataLoader(SequenceDataset(X_seq, np.zeros(len(X_seq))),
                        batch_size=512, shuffle=False, num_workers=2)
        return np.concatenate([model(Xb.to(DEVICE)).cpu().numpy() for Xb, _ in dl])


# ── Focal loss (XGBoost) ──────────────────────────────────────────────────────

def focal_loss_obj(y_pred, dtrain, gamma=2.0, alpha=0.25):
    y     = dtrain.get_label()
    p     = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -20, 20)))
    pt    = np.where(y == 1, p, 1 - p)
    at    = np.where(y == 1, alpha, 1 - alpha)
    fw    = at * (1 - pt) ** gamma
    grad  = fw * (p - y)
    hess  = fw * p * (1 - p)
    return grad, hess


# ── Stacking meta-learner ─────────────────────────────────────────────────────

def _load_regime_models(global_lgbm):
    models = {}
    expected = global_lgbm.num_feature()
    for s in range(4):
        p2c = os.path.join(MODELS, f'lgbm2c_regime_{s}.txt')
        p1  = os.path.join(MODELS, f'lgbm_regime_{s}.txt')
        p   = p2c if os.path.exists(p2c) else p1
        if os.path.exists(p):
            m = lgb.Booster(model_file=p)
            if m.num_feature() == expected:
                models[s] = m
            else:
                print(f"    Skipping regime_{s}: {m.num_feature()} features ≠ {expected}")
    return models


def _safe_regime_predict(model, global_lgbm, X_sub):
    """Predict with regime model; fall back to global on feature-count mismatch."""
    try:
        return model.predict(X_sub)
    except Exception:
        return global_lgbm.predict(X_sub)


def _soft_blend(X_tab, dates, posterior_by_date, regime_models, global_lgbm):
    if not regime_models or not posterior_by_date:
        return global_lgbm.predict(X_tab)
    preds = np.stack(
        [_safe_regime_predict(regime_models.get(s, global_lgbm), global_lgbm, X_tab)
         for s in range(4)], axis=1)
    posts = np.array(
        [posterior_by_date.get(str(d)[:10], [0.25]*4) for d in dates], dtype=np.float32)
    return (posts * preds).sum(axis=1)


def _hard_route(X_tab, dates, state_by_date, regime_models, global_lgbm):
    states = np.array([state_by_date.get(str(d)[:10], -1) for d in dates])
    probs  = np.zeros(len(X_tab), dtype=np.float32)
    for s in range(4):
        mask = states == s
        if not mask.any(): continue
        probs[mask] = _safe_regime_predict(
            regime_models.get(s, global_lgbm), global_lgbm, X_tab[mask])
    probs[states == -1] = global_lgbm.predict(X_tab[states == -1])
    return probs


def train_stacking(lgbm_model, lstm_model, X_tab, X_seq, y, df_sig):
    regime_models = _load_regime_models(lgbm_model)
    state_by_date, posterior_by_date = {}, {}
    hmm_path = os.path.join(MODELS, 'hmm_params.json')
    if os.path.exists(hmm_path):
        with open(hmm_path) as f:
            state_by_date = json.load(f).get('state_by_date', {})
    post_path = os.path.join(MODELS, 'hmm_posteriors.json')
    if os.path.exists(post_path):
        with open(post_path) as f:
            posterior_by_date = json.load(f)

    use_regime = bool(regime_models and state_by_date)
    use_soft   = bool(regime_models and posterior_by_date)
    print(f"  Regime models: {len(regime_models)}  |  soft blend: {use_soft}")

    dates   = df_sig['date'].values
    n       = len(y)
    xgb_oos = np.zeros(n)
    lgb_oos = np.zeros(n)
    lst_oos = np.zeros(n)
    reg_oos = np.zeros(n)
    sft_oos = np.zeros(n)
    # Sprint 4: Phase 2b/2c scores are pre-computed — use directly (no OOS leakage)
    p2b_oos = df_sig['lgbm2b_score'].values.astype(np.float32)
    p2c_oos = df_sig['lgbm2c_score'].values.astype(np.float32)

    splitter = PurgedTimeSeriesSplit(n_splits=5, embargo_days=7)

    for fold, (tr_idx, val_idx) in enumerate(splitter.split(X_tab, dates=dates)):
        print(f"    Fold {fold+1}/5  train={len(tr_idx)}  val={len(val_idx)}")
        Xtr_t, Xval_t = X_tab[tr_idx], X_tab[val_idx]
        Xtr_s, Xval_s = X_seq[tr_idx], X_seq[val_idx]
        ytr, yval     = y[tr_idx], y[val_idx]

        # XGBoost with focal loss
        dtrain = xgb.DMatrix(Xtr_t.astype(np.float32), label=ytr)
        dval   = xgb.DMatrix(Xval_t.astype(np.float32))
        m_xgb  = xgb.train(
            {'max_depth': 5, 'eta': 0.1, 'seed': 42,
             'monotone_constraints': tuple(MONOTONE_CONSTRAINTS)},
            dtrain, num_boost_round=200, obj=focal_loss_obj)
        xgb_oos[val_idx] = 1.0 / (1.0 + np.exp(-m_xgb.predict(dval)))

        # LightGBM (global)
        m_lgb = lgb.train({'objective': 'binary', 'verbose': -1, 'seed': 42,
                           'num_leaves': 31, 'learning_rate': 0.05},
                          lgb.Dataset(Xtr_t, label=ytr), num_boost_round=200)
        lgb_oos[val_idx] = m_lgb.predict(Xval_t)

        # LSTM OOS
        lst_oos[val_idx] = get_lstm_probs(lstm_model, Xval_s)

        # Regime routing
        if use_regime:
            reg_oos[val_idx] = _hard_route(
                Xval_t, dates[val_idx], state_by_date, regime_models, lgbm_model)
        else:
            reg_oos[val_idx] = lgb_oos[val_idx]

        if use_soft:
            sft_oos[val_idx] = _soft_blend(
                Xval_t, dates[val_idx], posterior_by_date, regime_models, lgbm_model)
        else:
            sft_oos[val_idx] = reg_oos[val_idx]

    # Optuna tune XGBoost on fold-1
    print("\n  Tuning stacking XGB (15 trials) ...")
    fold1_tr, fold1_val = next(iter(
        PurgedTimeSeriesSplit(n_splits=5, embargo_days=7).split(X_tab, dates=dates)))
    dtrain_t = xgb.DMatrix(X_tab[fold1_tr].astype(np.float32), label=y[fold1_tr])
    dval_t   = xgb.DMatrix(X_tab[fold1_val].astype(np.float32))

    def _xgb_obj(trial):
        p = {'max_depth': trial.suggest_int('max_depth', 3, 7),
             'eta':       trial.suggest_float('eta', 0.03, 0.2, log=True),
             'subsample': trial.suggest_float('subsample', 0.6, 1.0),
             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
             'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
             'seed': 42, 'monotone_constraints': tuple(MONOTONE_CONSTRAINTS)}
        m   = xgb.train(p, dtrain_t, num_boost_round=200, obj=focal_loss_obj)
        raw = m.predict(dval_t)
        return roc_auc_score(y[fold1_val], 1.0 / (1.0 + np.exp(-raw)))

    try:
        import optuna as _optuna
        _optuna.logging.set_verbosity(_optuna.logging.WARNING)
        study = _optuna.create_study(direction='maximize',
                                     sampler=_optuna.samplers.TPESampler(seed=42))
        study.optimize(_xgb_obj, n_trials=15, show_progress_bar=False)
        best_xgb_params = {**study.best_params, 'seed': 42,
                           'monotone_constraints': tuple(MONOTONE_CONSTRAINTS)}
        print(f"  Best XGB AUC={study.best_value:.4f}")
    except Exception as e:
        print(f"  XGB Optuna skipped ({e}) — default params")
        best_xgb_params = {'max_depth': 5, 'eta': 0.1, 'seed': 42,
                           'monotone_constraints': tuple(MONOTONE_CONSTRAINTS)}

    # Re-run all 5 folds with tuned XGB
    xgb_tuned = np.zeros(n)
    for fold, (tr_idx, val_idx) in enumerate(
            PurgedTimeSeriesSplit(n_splits=5, embargo_days=7).split(X_tab, dates=dates)):
        dtf = xgb.DMatrix(X_tab[tr_idx].astype(np.float32), label=y[tr_idx])
        dvf = xgb.DMatrix(X_tab[val_idx].astype(np.float32))
        mf  = xgb.train(best_xgb_params, dtf, num_boost_round=200, obj=focal_loss_obj)
        xgb_tuned[val_idx] = 1.0 / (1.0 + np.exp(-mf.predict(dvf)))
    xgb_oos = xgb_tuned

    # LightGBM meta-learner (Sprint 4: +2 Phase 2b/2c columns = 7 base learners)
    print("  Fitting LightGBM meta-learner (7 base learners: xgb/lgb/lstm/regime/soft/p2b/p2c) ...")
    meta_X    = np.column_stack([xgb_oos, lgb_oos, lst_oos, reg_oos, sft_oos, p2b_oos, p2c_oos])
    n_meta_tr = int(n * 0.80)
    meta_tr   = lgb.Dataset(meta_X[:n_meta_tr], label=y[:n_meta_tr])
    meta_val  = lgb.Dataset(meta_X[n_meta_tr:], label=y[n_meta_tr:], reference=meta_tr)
    meta_lgb  = lgb.train(
        {'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'seed': 42,
         'num_leaves': 8, 'max_depth': 3, 'learning_rate': 0.05,
         'feature_fraction': 1.0, 'bagging_fraction': 0.8, 'bagging_freq': 5,
         'min_child_samples': 50},
        meta_tr, num_boost_round=300, valid_sets=[meta_val],
        callbacks=[lgb.early_stopping(30, verbose=False)])
    out_meta = os.path.join(WORK, 'meta_lgbm.txt')
    meta_lgb.save_model(out_meta)
    print(f"  Meta-LightGBM → {out_meta}")

    oos_preds  = meta_lgb.predict(meta_X[n_meta_tr:])
    auc_meta   = roc_auc_score(y[n_meta_tr:], oos_preds)
    full_preds = meta_lgb.predict(meta_X)

    auc_xgb  = roc_auc_score(y, xgb_oos)
    auc_lgb  = roc_auc_score(y, lgb_oos)
    auc_lst  = roc_auc_score(y, lst_oos)
    auc_reg  = roc_auc_score(y, reg_oos)
    auc_sft  = roc_auc_score(y, sft_oos)
    auc_p2b  = roc_auc_score(y, p2b_oos)
    auc_p2c  = roc_auc_score(y, p2c_oos)
    print(f"\n  OOS AUC — XGB:{auc_xgb:.4f}  LGB:{auc_lgb:.4f}  LSTM:{auc_lst:.4f}"
          f"  Regime:{auc_reg:.4f}  Soft:{auc_sft:.4f}"
          f"  P2b:{auc_p2b:.4f}  P2c:{auc_p2c:.4f}  STACK:{auc_meta:.4f}")

    # Threshold calibration on held-out slice
    y_cal = y[n_meta_tr:]
    best_thresh, best_prec = 0.5, 0.0
    for t in np.arange(0.30, 0.80, 0.01):
        pb  = (oos_preds >= t).astype(int)
        tp  = int((pb & y_cal).sum())
        fp  = int((pb & (1 - y_cal)).sum())
        fn  = int(((1 - pb) & y_cal).sum())
        pr  = tp / max(tp + fp, 1)
        rc  = tp / max(tp + fn, 1)
        if rc >= 0.30 and pr > best_prec:
            best_prec, best_thresh = pr, t
    print(f"  Threshold: {best_thresh:.2f}  (precision={best_prec:.3f} at recall≥0.30)")

    return {
        'meta_learner':  'lgbm',
        'meta_learner_inputs': ['xgb', 'lgbm', 'lstm', 'regime', 'soft', 'p2b', 'p2c'],
        'meta_lgbm_path': out_meta,
        'auc_xgb':     round(float(auc_xgb),  4),
        'auc_lgbm':    round(float(auc_lgb),   4),
        'auc_lstm':    round(float(auc_lst),   4),
        'auc_regime':  round(float(auc_reg),   4),
        'auc_soft':    round(float(auc_sft),   4),
        'auc_p2b':     round(float(auc_p2b),   4),
        'auc_p2c':     round(float(auc_p2c),   4),
        'auc_stack':   round(float(auc_meta),  4),
        'threshold':   round(float(best_thresh), 2),
        'threshold_precision': round(float(best_prec), 4),
        'best_xgb_params': {k: v for k, v in best_xgb_params.items()
                            if k != 'monotone_constraints'},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CPR Phase 3: LSTM + Stacking  (Kaggle GPU)")
    print("=" * 60)
    print(f"  Device: {DEVICE}  |  EPOCHS={EPOCHS}  BATCH={BATCH_SIZE}")

    _lgbm_p2c = os.path.join(MODELS, 'lgbm2c_global.txt')
    _lgbm_p1  = os.path.join(MODELS, 'lgbm_model.txt')
    _lgbm_path = _lgbm_p2c if os.path.exists(_lgbm_p2c) else _lgbm_p1
    print(f"  Loading LightGBM from: {os.path.basename(_lgbm_path)}")
    lgbm_model = lgb.Booster(model_file=_lgbm_path)
    df_sig     = load_signal_dataset()

    # Cap at 400k for sequence building (10-feature seqs × 400K × 30 bars ≈ 480 MB)
    if len(df_sig) > 400_000:
        df_sig = df_sig.sample(400_000, random_state=42).sort_values('date').reset_index(drop=True)
        print(f"  Sampled 400k signals for LSTM.")

    # Compute Phase 2c interaction features (required for 46-feature models)
    df_sig['cpr_vol_interaction']    = df_sig['cpr_compress'] * df_sig['vol_rank']
    df_sig['regime_momentum']        = df_sig['hmm_regime']   * df_sig['mom5']
    df_sig['cpr_rsi_squeeze']        = (1 - df_sig['cpr_width_pct']) * df_sig['rsi14']
    df_sig['overlap_vol_signal']     = df_sig['cpr_overlap_pct'] * df_sig['cpr_zone_vol_ratio']
    df_sig['rs_direction_alignment'] = (df_sig['market_rs_5d'] + df_sig['sector_rs_5d']) * df_sig['direction']
    df_sig['virgin_momentum']        = df_sig.get('cpr_virgin', 0.0) * df_sig.get('mom5', 0.0)
    df_sig['narrow_breakout_vol']    = df_sig.get('consecutive_narrow_cprs', 0.0) * df_sig.get('vol_rank', 1.0)

    X_tab = df_sig[FEATURE_COLS].fillna(0).values.astype(np.float32)
    y     = df_sig[WIN_COL].values.astype(int)

    # Build sequences
    print("\n── Phase 3A: Building 30-bar sequences ─────────────────────────")
    ohlcv_path = os.path.join(INPUT, 'ohlcv', 'ALL_SYMBOLS_OHLCV.csv')
    if os.path.exists(ohlcv_path):
        df_all = pd.read_csv(ohlcv_path, low_memory=False)
        df_all.columns = [c.strip().title() for c in df_all.columns]
        df_all['Date'] = pd.to_datetime(df_all['Date'], dayfirst=True)
        X_seq, y_seq, valid_pos = build_sequences(df_all, df_sig)
        print(f"  Built {len(X_seq)} sequences from OHLCV data.")
        df_sig = df_sig.iloc[valid_pos].reset_index(drop=True)
        X_tab  = df_sig[FEATURE_COLS].values.astype(np.float32)
        y      = y_seq
    else:
        print("  OHLCV not found — using tabular proxy sequences (10 features, padded/repeated).")
        seq_feat = np.zeros((len(X_tab), len(SEQUENCE_COLS)), dtype=np.float32)
        n_base   = min(6, X_tab.shape[1])
        seq_feat[:, :n_base] = X_tab[:, :n_base]
        X_seq    = np.tile(seq_feat[:, np.newaxis, :], (1, SEQ_LEN, 1))

    # Temporal split 80/20
    n_tr              = int(len(y) * 0.80)
    X_t_tr, X_t_te   = X_tab[:n_tr], X_tab[n_tr:]
    X_s_tr, X_s_te   = X_seq[:n_tr], X_seq[n_tr:]
    y_tr,   y_te     = y[:n_tr],     y[n_tr:]

    # Train LSTM
    print("\n── Phase 3A: Training LSTM ──────────────────────────────────────")
    lstm_model, lstm_auc = train_lstm(X_s_tr, y_tr, X_s_te, y_te)

    out_lstm = os.path.join(WORK, 'lstm_model.pt')
    torch.save({
        'model_state': lstm_model.state_dict(),
        'input_dim':   len(SEQUENCE_COLS),
        'hidden_dim':  HIDDEN_DIM,
        'seq_len':     SEQ_LEN,
        'seq_cols':    SEQUENCE_COLS,
    }, out_lstm)
    print(f"  Saved → {out_lstm}")

    # Ensure lgbm_model feature count matches X_tab (stale models have fewer features)
    if lgbm_model.num_feature() != X_tab.shape[1]:
        print(f"  ⚠ Loaded lgbm has {lgbm_model.num_feature()} features, X_tab has {X_tab.shape[1]}.")
        print("    Retraining lgbm on current feature set for stacking ...")
        dtrain_full = lgb.Dataset(X_tab, label=y)
        lgbm_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 127,
             'learning_rate': 0.05, 'min_child_samples': 20,
             'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
             'verbose': -1, 'n_jobs': -1},
            dtrain_full, num_boost_round=300,
        )
        # Save refreshed model so download_outputs picks it up
        lgbm_model.save_model(os.path.join(WORK, 'meta_lgbm.txt'))
        print("    Retrained lgbm saved → meta_lgbm.txt")

    # Stacking
    print("\n── Phase 3B: Stacking Meta-Learner ─────────────────────────────")
    stacking_weights = train_stacking(lgbm_model, lstm_model, X_tab, X_seq, y, df_sig)

    out_sw = os.path.join(WORK, 'stacking_weights.json')
    with open(out_sw, 'w') as f:
        json.dump(stacking_weights, f, indent=2)
    print(f"  Saved → {out_sw}")

    metrics = {'lstm_auc': round(float(lstm_auc), 4), **stacking_weights}
    with open(os.path.join(WORK, 'phase3_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("  Phase 3 complete. Outputs in /kaggle/working/")
    print("=" * 60)


if __name__ == '__main__':
    main()
