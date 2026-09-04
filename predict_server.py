"""
Multi-model prediction microservice — port 5001

Endpoints:
  POST /predict              → XGBoost score (original, backward-compatible)
  POST /predict_ensemble     → XGB + LGBM + LSTM stacking ensemble score
  GET  /regime               → current HMM regime
  POST /position_size        → PPO position sizing recommendation
  GET  /gate_weights         → SHAP-derived gate weights
  GET  /health
  POST /api/train/start      → launch run_all.py pipeline
  GET  /api/train/status     → current pipeline status + last-run time
  GET  /api/train/stream     → SSE stream of pipeline stdout
  POST /api/train/cancel     → kill running pipeline
"""

import os, json, sys, threading, subprocess, time, queue
import numpy as np
from flask import Flask, request, jsonify, Response
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.ml.scoring import conformal_interval, stacking_ensemble, ppo_position_size

app = Flask(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE, 'models')
XGB_PATH    = os.path.join(MODELS_DIR, 'xgb_phase2.json')
XGB_FALLBACK= os.path.join(MODELS_DIR, 'xgb_regressor_v2.json')
LGBM_PATH   = os.path.join(MODELS_DIR, 'lgbm_model.txt')
META_LGBM_PATH = os.path.join(MODELS_DIR, 'meta_lgbm.txt')
REGIME_PATHS   = {s: os.path.join(MODELS_DIR, f'lgbm_regime_{s}.txt') for s in range(4)}
RULE_PATHS     = {i: os.path.join(MODELS_DIR, f'lgbm_rule{i}.txt') for i in range(1, 12)}
LSTM_PATH   = os.path.join(MODELS_DIR, 'lstm_model.pt')
HMM_PATH    = os.path.join(MODELS_DIR, 'hmm_params.json')
POSTERIORS_PATH = os.path.join(MODELS_DIR, 'hmm_posteriors.json')
STACKING_PATH  = os.path.join(MODELS_DIR, 'stacking_weights.json')
PPO_PATH    = os.path.join(MODELS_DIR, 'ppo_policy_weights.json')
GATE_W_PATH = os.path.join(MODELS_DIR, 'shap_gate_weights.json')
CONFORMAL_PATH = os.path.join(MODELS_DIR, 'conformal_scores.json')
LGBM_SCORER_PATH  = os.path.join(MODELS_DIR, 'lgbm_scorer.txt')
SHAP_WEIGHTS_PATH = os.path.join(MODELS_DIR, 'shap_weights.json')
# Phase 2c — regime-conditional signal scorer
LGBM2C_GLOBAL_PATH  = os.path.join(MODELS_DIR, 'lgbm2c_global.txt')
LGBM2C_REGIME_PATHS = {s: os.path.join(MODELS_DIR, f'lgbm2c_regime_{s}.txt') for s in range(4)}
SHAP_WEIGHTS2C_PATH = os.path.join(MODELS_DIR, 'shap_weights2c.json')

FEATURES = [
    # original 12
    'cpr_width_pct', 'vwap_dist', 'atr_pct_rank', 'vol_rank',
    'n_rules_fired', 'sg_vel', 'ema200_dist', 'rsi14',
    'mom5', 'dow', 'rule_id', 'direction',
    # Phase A: 52-week context + volume acceleration
    'dist_hi52', 'dist_lo52', 'vol_accel',
    # Phase B: relative strength (market + sector)
    'market_rs_5d', 'market_rs_20d', 'sector_rs_5d', 'sector_rs_20d',
    # Phase C: delivery %
    'deliv_pct',
    # Phase D: put-call ratio
    'pcr',
    # Tier 1: VIX + interaction features
    'india_vix', 'conf_vol', 'rsi_dir', 'hi52_dir',
    # Tier 2A: CPR quality
    'cpr_compress', 'cpr_pos', 'dist_r1', 'dist_s1',
    # Tier 2B: multi-timeframe momentum
    'mom3', 'mom10', 'mom20',
    # Tier 2C: divergence + volume curvature
    'rsi_div', 'vol_accel_delta',
    # Tier 2D: context
    'days_since_52hi', 'expiry_dist',
    # HMM regime quality
    'regime_stability', 'transition_risk',
]

# ─── Model Registry ───────────────────────────────────────────────────────────
models = {}
lock   = threading.Lock()


def load_models():
    global models
    m = {}

    # XGBoost (always required)
    print("Loading XGBoost…")
    xgb_p = XGB_PATH if os.path.exists(XGB_PATH) else XGB_FALLBACK
    xgb_m = xgb.XGBRegressor()
    xgb_m.load_model(xgb_p)
    m['xgb'] = xgb_m
    print(f"  XGBoost ready ({os.path.basename(xgb_p)}).")

    # LightGBM
    if os.path.exists(LGBM_PATH):
        try:
            import lightgbm as lgb
            m['lgbm'] = lgb.Booster(model_file=LGBM_PATH)
            print("  LightGBM ready.")
        except Exception as e:
            print(f"  LightGBM skip: {e}")

    # LSTM (PyTorch)
    if os.path.exists(LSTM_PATH):
        try:
            import torch
            ckpt = torch.load(LSTM_PATH, map_location='cpu', weights_only=False)
            from scripts.ml.lstm_model import LSTMSignalModel
            lstm = LSTMSignalModel(
                input_dim=len(ckpt['seq_cols']),
                hidden_dim=ckpt['hidden_dim']
            )
            lstm.load_state_dict(ckpt['model_state'])
            lstm.eval()
            m['lstm'] = lstm
            m['lstm_seq_len'] = ckpt['seq_len']
            m['lstm_seq_cols'] = ckpt['seq_cols']
            print("  LSTM ready.")
        except Exception as e:
            print(f"  LSTM skip: {e}")

    # Regime sub-models (LightGBM per HMM state)
    try:
        import lightgbm as lgb_mod
        regime_models = {}
        for s, path in REGIME_PATHS.items():
            if os.path.exists(path):
                regime_models[s] = lgb_mod.Booster(model_file=path)
        if regime_models:
            m['regime_models'] = regime_models
            print(f"  Regime sub-models ready: states {sorted(regime_models.keys())}")
    except Exception as e:
        print(f"  Regime sub-models skip: {e}")

    # Per-rule sub-models (LightGBM per rule1..11)
    try:
        import lightgbm as lgb_mod
        rule_models = {}
        for rule_num, path in RULE_PATHS.items():
            if os.path.exists(path):
                rule_models[rule_num] = lgb_mod.Booster(model_file=path)
        if rule_models:
            m['rule_models'] = rule_models
            print(f"  Per-rule sub-models ready: rules {sorted(rule_models.keys())}")
    except Exception as e:
        print(f"  Per-rule sub-models skip: {e}")

    # Meta-LightGBM stacking learner (produced by Phase 3 retrain)
    if os.path.exists(META_LGBM_PATH):
        try:
            import lightgbm as lgb_mod
            m['meta_lgbm'] = lgb_mod.Booster(model_file=META_LGBM_PATH)
            print("  Meta-LightGBM ready.")
        except Exception as e:
            print(f"  Meta-LightGBM skip: {e}")

    # HMM params (JSON, decoded in JS; here for regime query)
    if os.path.exists(HMM_PATH):
        with open(HMM_PATH) as f:
            m['hmm'] = json.load(f)
        print("  HMM params ready.")

    # HMM posteriors — use last date's posterior as current regime distribution
    if os.path.exists(POSTERIORS_PATH):
        with open(POSTERIORS_PATH) as f:
            posteriors = json.load(f)
        if posteriors:
            last_date = sorted(posteriors.keys())[-1]
            m['current_posterior'] = posteriors[last_date]  # list of 4 floats
            print(f"  HMM posteriors ready (last date: {last_date}).")

    # Stacking weights (logistic regression fallback)
    if os.path.exists(STACKING_PATH):
        with open(STACKING_PATH) as f:
            m['stacking'] = json.load(f)
        print("  Stacking weights ready.")

    # PPO policy weights
    if os.path.exists(PPO_PATH):
        with open(PPO_PATH) as f:
            m['ppo_weights'] = json.load(f)
        print("  PPO weights ready.")

    # Phase 2b LightGBM direct signal scorer
    if os.path.exists(LGBM_SCORER_PATH):
        try:
            import lightgbm as lgb_mod
            m['lgbm_scorer'] = lgb_mod.Booster(model_file=LGBM_SCORER_PATH)
            print("  Phase2b LGBM scorer ready.")
        except Exception as e:
            print(f"  Phase2b LGBM scorer skip: {e}")

    # Phase 2b SHAP feature weights
    if os.path.exists(SHAP_WEIGHTS_PATH):
        with open(SHAP_WEIGHTS_PATH) as f:
            m['shap_weights'] = json.load(f)
        print("  Phase2b SHAP weights ready.")

    # Phase 2c — regime-conditional LightGBM scorer (global + per-regime)
    try:
        import lightgbm as lgb_mod
        if os.path.exists(LGBM2C_GLOBAL_PATH):
            m['lgbm2c_global'] = lgb_mod.Booster(model_file=LGBM2C_GLOBAL_PATH)
            print("  Phase2c global scorer ready.")
        p2c_regime = {}
        for s, path in LGBM2C_REGIME_PATHS.items():
            if os.path.exists(path):
                p2c_regime[s] = lgb_mod.Booster(model_file=path)
        if p2c_regime:
            m['lgbm2c_regimes'] = p2c_regime
            print(f"  Phase2c regime scorers ready: {sorted(p2c_regime.keys())}")
    except Exception as e:
        print(f"  Phase2c scorers skip: {e}")

    if os.path.exists(SHAP_WEIGHTS2C_PATH):
        with open(SHAP_WEIGHTS2C_PATH) as f:
            m['shap_weights2c'] = json.load(f)
        print("  Phase2c SHAP weights ready.")

    # SHAP gate weights
    if os.path.exists(GATE_W_PATH):
        with open(GATE_W_PATH) as f:
            m['gate_weights'] = json.load(f)
        print("  Gate weights ready.")

    # Conformal scores
    if os.path.exists(CONFORMAL_PATH):
        with open(CONFORMAL_PATH) as f:
            m['conformal'] = json.load(f)
        print("  Conformal scores ready.")

    with lock:
        models.update(m)
    print("All models loaded.")


# ─────────────────────────── Helpers ─────────────────────────────────────────

_FEATURE_DEFAULTS = {
    'cpr_width_pct': 0.0,
    'vwap_dist':     0.0,
    'atr_pct_rank':  0.5,
    'vol_rank':      0.5,
    'n_rules_fired': 1.0,
    'sg_vel':        0.0,
    'ema200_dist':   0.0,
    'rsi14':        50.0,
    'mom5':          0.0,
    'dow':           2.0,
    'rule_id':       1.0,
    'direction':     1.0,
    'dist_hi52':    -0.1,
    'dist_lo52':     0.1,
    'vol_accel':     1.0,
    'market_rs_5d':  1.0,
    'market_rs_20d': 1.0,
    'sector_rs_5d':  1.0,
    'sector_rs_20d': 1.0,
    'deliv_pct':     0.0,
    'pcr':           1.0,
    'india_vix':    15.0,
    'conf_vol':      2.0,
    'rsi_dir':       0.0,
    'hi52_dir':      0.0,
    # Tier 2A: CPR quality
    'cpr_compress':  1.0,   # neutral = today CPR = 5d avg
    'cpr_pos':       0.5,   # close at midpoint of CPR
    'dist_r1':      -0.02,  # close ~2% below R1
    'dist_s1':       0.02,  # close ~2% above S1
    # Tier 2B: multi-timeframe momentum
    'mom3':          0.0,
    'mom10':         0.0,
    'mom20':         0.0,
    # Tier 2C: divergence + volume curvature
    'rsi_div':       0.0,
    'vol_accel_delta': 0.0,
    # Tier 2D: context
    'days_since_52hi': 90.0,
    'expiry_dist':   15.0,
    # HMM regime quality
    'regime_stability': 0.0,
    'transition_risk':  0.25,
}


def extract_features(data):
    if isinstance(data, dict):
        data = [data]
    rows = [[float(d.get(f, _FEATURE_DEFAULTS.get(f, 0.0))) for f in FEATURES] for d in data]
    return np.array(rows, dtype=np.float32)


# ── Phase 2c feature extraction ───────────────────────────────────────────────

_P2C_BASE_FEATURES = [
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
    # Sprint 1 CPR features
    'cpr_overlap_pct', 'open_to_cpr_dist', 'prev_cpr_respected', 'cpr_zone_vol_ratio',
    # HMM regime
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
]  # 56

_P2C_INTERACTION_FEATURES = [
    'cpr_vol_interaction',
    'regime_momentum',
    'cpr_rsi_squeeze',
    'overlap_vol_signal',
    'rs_direction_alignment',
    'virgin_momentum',
    'narrow_breakout_vol',
]  # 7

_P2C_ALL_FEATURES = _P2C_BASE_FEATURES + _P2C_INTERACTION_FEATURES  # 63

# Directional features: sign-flip for SELL (direction=-1). Matches score_p2c.py.
_P2C_DIRECTIONAL = {
    'dist_hi52', 'dist_lo52', 'vwap_dist', 'ema200_dist',
    'mom3', 'mom5', 'mom10', 'mom20',
    'market_rs_5d', 'market_rs_20d', 'sector_rs_5d', 'sector_rs_20d',
    'cpr_pos', 'dist_r1', 'dist_s1', 'sg_vel',
    'open_to_cpr_dist',
    'gap_pct',
}

_P2C_DEFAULTS = {
    'cpr_overlap_pct':           0.5,
    'open_to_cpr_dist':          0.0,
    'prev_cpr_respected':        0.0,
    'cpr_zone_vol_ratio':        1.0,
    'hmm_regime':               -1,
    # Sprint 2A
    'open_inside_cpr':           0.0,
    'cpr_virgin':                0.0,
    'consecutive_narrow_cprs':   0.0,
    'cpr_midpoint_trend':        0.0,
    'cpr_expansion_factor':      1.0,
    # Sprint 2B
    'cpr_above_prev_cpr':        0.0,
    'prev_close_inside_cpr':     0.0,
    'atr_to_cpr_ratio':          1.0,
    'cpr_width_percentile_252d': 0.5,
    'prev_day_ochoa_type':       0.0,
    # Sprint 3
    'gap_pct':                   0.0,
    'cpr_test_count_5d':         0.0,
    'prev_bar_close_pos':        0.5,
    'atr_expansion':             1.0,
    'vol_trend_slope':           0.0,
}


def _p2c_current_regime():
    """Return current HMM regime as int (argmax of latest posterior, or -1)."""
    posterior = models.get('current_posterior')
    if posterior:
        return int(np.argmax(posterior))
    return -1


def extract_features_2c(data):
    """Build 63-feature array for Phase 2c models. Matches score_p2c.py logic exactly."""
    if isinstance(data, dict):
        data = [data]

    current_regime = _p2c_current_regime()
    rows = []
    for d in data:
        vals = {}
        for f in _P2C_BASE_FEATURES:
            default = _P2C_DEFAULTS.get(f, _FEATURE_DEFAULTS.get(f, 0.0))
            vals[f] = float(d.get(f, default))

        if vals['hmm_regime'] == -1 and current_regime >= 0:
            vals['hmm_regime'] = float(current_regime)

        # Direction-adjust signed features (SELL signals get sign-flipped)
        direction = vals['direction']
        if direction == -1:
            for col in _P2C_DIRECTIONAL:
                if col in vals:
                    vals[col] = vals[col] * -1

        # Interaction features — exact formulas from score_p2c.py
        vals['cpr_vol_interaction']    = vals['cpr_compress'] * vals['vol_rank']
        vals['regime_momentum']        = vals['hmm_regime'] * vals['mom5']
        vals['cpr_rsi_squeeze']        = (1.0 - vals['cpr_width_pct']) * vals['rsi14']
        vals['overlap_vol_signal']     = vals['cpr_overlap_pct'] * vals['cpr_zone_vol_ratio']
        vals['rs_direction_alignment'] = (vals['market_rs_5d'] + vals['sector_rs_5d']) * direction
        vals['virgin_momentum']        = vals['cpr_virgin'] * vals['mom5']
        vals['narrow_breakout_vol']    = vals['consecutive_narrow_cprs'] * vals['vol_rank']

        rows.append([float(np.nan_to_num(vals.get(f, 0.0))) for f in _P2C_ALL_FEATURES])

    return np.array(rows, dtype=np.float32)


def score_lgbm2c(X_2c, hmm_regimes=None):
    """50% global + 50% regime-specific blend. Matches score_p2c.py scoring."""
    global_m   = models.get('lgbm2c_global')
    regime_map = models.get('lgbm2c_regimes', {})

    if global_m is None:
        return None

    global_scores = global_m.predict(X_2c)
    final_scores  = global_scores.copy()

    current_regime = _p2c_current_regime()
    regime_m = regime_map.get(current_regime)
    if regime_m is not None and regime_m.num_feature() == X_2c.shape[1]:
        regime_scores = regime_m.predict(X_2c)
        final_scores  = 0.5 * global_scores + 0.5 * regime_scores

    return np.clip(final_scores, 0.0, 1.0).astype(np.float32)


def _conformal(raw_score: float, alpha: float = 0.10):
    return conformal_interval(raw_score, models.get('conformal'), alpha)


def _stacking(xgb_prob, lgbm_prob, lstm_prob=None, regime_hard=None, soft_blend=None,
              p2b_prob=None, p2c_prob=None):
    return stacking_ensemble(
        xgb_prob, lgbm_prob,
        meta_lgbm=models.get('meta_lgbm'),
        stacking_weights=models.get('stacking'),
        lstm_prob=lstm_prob,
        regime_hard=regime_hard,
        soft_blend=soft_blend,
        p2b_prob=p2b_prob,
        p2c_prob=p2c_prob,
    )


def _ppo(features_row, regime_score: float = 0.5):
    return ppo_position_size(features_row, models.get('ppo_weights'), regime_score)


# ─────────────────────────── Routes ──────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    try:
        X = extract_features(request.json)
        preds = models['xgb'].predict(X).tolist()
        return jsonify({'predictions': preds})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict_ensemble', methods=['POST'])
def predict_ensemble():
    try:
        data = request.json
        X = extract_features(data)

        xgb_preds  = models['xgb'].predict(X)
        lgbm_m     = models.get('lgbm')
        lgbm_preds = lgbm_m.predict(X) if lgbm_m else xgb_preds

        # Per-rule blend: 70% global LGBM + 30% rule-specific model (if available)
        rule_models = models.get('rule_models', {})
        if rule_models and lgbm_m is not None:
            input_list = data if isinstance(data, list) else [data]
            rule_preds = lgbm_preds.copy()
            for j, d in enumerate(input_list):
                rid = int(d.get('rule_id', _FEATURE_DEFAULTS['rule_id']))
                rm = rule_models.get(rid)
                if rm is not None:
                    rule_preds[j] = 0.7 * lgbm_preds[j] + 0.3 * float(rm.predict(X[j:j+1])[0])
            lgbm_preds = rule_preds

        # Regime routing
        hmm_state     = models.get('hmm', {}).get('current_state', -1)
        regime_models = models.get('regime_models', {})
        current_regime_m = regime_models.get(hmm_state)
        regime_hard_preds = (current_regime_m.predict(X) if current_regime_m
                             else lgbm_preds)

        # Soft blend: weighted average of all regime sub-models by current posterior
        posterior = models.get('current_posterior', [0.25, 0.25, 0.25, 0.25])
        if regime_models and lgbm_m is not None:
            all_regime = np.stack(
                [regime_models.get(s, lgbm_m).predict(X) for s in range(4)],
                axis=1,
            )  # [N, 4]
            soft_preds = (all_regime * np.array(posterior, dtype=np.float32)).sum(axis=1)
        else:
            soft_preds = lgbm_preds

        # Phase 2c regime-conditional scoring
        X_2c      = extract_features_2c(data)
        p2c_preds = score_lgbm2c(X_2c)   # None if models not yet loaded

        results = []
        for i, row in enumerate(X):
            xgb_p   = float(np.clip(xgb_preds[i], 0, 1))
            lgbm_p  = float(np.clip(lgbm_preds[i], 0, 1))
            reg_h   = float(np.clip(regime_hard_preds[i], 0, 1))
            soft_v  = float(np.clip(soft_preds[i], 0, 1))
            p2c_v   = float(np.clip(p2c_preds[i], 0, 1)) if p2c_preds is not None else None
            # Phase 2c is primary scorer (AUC 0.6465 vs meta-stack 0.6116 on recent data)
            stack   = p2c_v if p2c_v is not None else _stacking(
                xgb_p, lgbm_p, regime_hard=reg_h, soft_blend=soft_v,
                p2b_prob=lgbm_p, p2c_prob=p2c_v)
            lo, hi  = _conformal(stack)

            entry = {
                'xgb_score':    round(xgb_p, 4),
                'lgbm_score':   round(lgbm_p, 4),
                'regime_score': round(reg_h, 4),
                'soft_score':   round(soft_v, 4),
                'stack_score':  round(stack, 4),
                'conf_lower':   round(lo, 4),
                'conf_upper':   round(hi, 4),
            }
            if p2c_preds is not None:
                entry['lgbm2c_score'] = round(p2c_v, 4)
                entry['lgbm2c_regime'] = _p2c_current_regime()

            results.append(entry)
        return jsonify({'ensemble': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/regime', methods=['GET'])
def regime():
    hmm = models.get('hmm')
    if not hmm:
        return jsonify({'regime': 'Unknown', 'available': False})
    return jsonify({
        'regime':        hmm.get('current_regime', 'Unknown'),
        'state':         hmm.get('current_state', -1),
        'regime_map':    hmm.get('regime_map', {}),
        'available':     True,
    })


@app.route('/position_size', methods=['POST'])
def position_size():
    try:
        data   = request.json
        # PPO was trained on 60-dim state: FEATURE_COLS(56) + [lgbm2c_score, exposure, cum_pnl, win_streak]
        # Use extract_features_2c to get correct 56-feature order, strip the 7 interaction cols
        X_2c   = extract_features_2c(data if isinstance(data, list) else [data])
        X      = X_2c[:, :len(_P2C_BASE_FEATURES)]   # 56 base features only
        regime = request.args.get('regime', 'Bull-Trend')
        regime_score = {'Bull-Trend': 1.0, 'Bear-Trend': 0.4,
                        'Chop': 0.2, 'High-Vol-Panic': 0.0}.get(regime, 0.6)
        sizes  = [_ppo(row, regime_score) for row in X]
        return jsonify({'position_sizes': sizes, 'regime_score': regime_score})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/gate_weights', methods=['GET'])
def gate_weights():
    gw = models.get('gate_weights')
    if not gw:
        return jsonify({'available': False, 'gate_weights': {}})
    return jsonify({'available': True, 'gate_weights': gw.get('gate_weights', {})})


@app.route('/health', methods=['GET'])
def health():
    loaded = [k for k in [
        'xgb', 'lgbm', 'lstm', 'hmm', 'stacking',
        'meta_lgbm', 'regime_models', 'current_posterior',
        'ppo_weights', 'gate_weights',
        'lgbm_scorer', 'lgbm2c_global', 'lgbm2c_regimes', 'shap_weights2c',
    ] if k in models]
    return jsonify({'status': 'ok', 'models_loaded': loaded})


# ─────────────────────────── Train pipeline routes ───────────────────────────

_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
_RUN_ALL     = os.path.join(_BASE_DIR, 'scripts', 'ml', 'run_all.py')
_TRAIN_STAMP = os.path.join(_BASE_DIR, 'auto_retrain_cpr.last_run')

_train_state = {
    'status':   'idle',    # idle | running | done | failed | cancelled
    'last_run': None,
    'proc':     None,
    'lock':     threading.Lock(),
    'queue':    queue.Queue(),
}


def _read_last_run():
    try:
        if os.path.exists(_TRAIN_STAMP):
            return open(_TRAIN_STAMP).read().strip()
    except Exception:
        pass
    return None


def _pipeline_thread(cmd):
    st = _train_state
    try:
        proc = subprocess.Popen(
            cmd, cwd=_BASE_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        with st['lock']:
            st['proc'] = proc

        for line in proc.stdout:
            st['queue'].put({'line': line.rstrip()})

        proc.wait()
        with st['lock']:
            st['proc'] = None
            if proc.returncode == 0:
                st['status']   = 'done'
                st['last_run'] = time.strftime('%Y-%m-%dT%H:%M:%S')
                # persist stamp
                try:
                    open(_TRAIN_STAMP, 'w').write(st['last_run'])
                except Exception:
                    pass
                load_models()   # hot-reload new model files
            elif st['status'] != 'cancelled':
                st['status'] = 'failed'
        st['queue'].put({'status': st['status']})

    except Exception as exc:
        with st['lock']:
            st['proc']   = None
            st['status'] = 'failed'
        _train_state['queue'].put({'line': f'[INTERNAL ERROR] {exc}', 'status': 'failed'})


@app.route('/api/train/start', methods=['POST'])
def train_start():
    st = _train_state
    with st['lock']:
        if st['status'] == 'running':
            return jsonify({'error': 'Pipeline already running'}), 409
        st['status'] = 'running'
        # drain stale queue
        while not st['queue'].empty():
            try:
                st['queue'].get_nowait()
            except queue.Empty:
                break

    skip_upload  = request.args.get('skipUpload') == '1'
    local_phase4 = request.args.get('localPhase4') == '1'

    cmd = [sys.executable, _RUN_ALL, '--skip-dataset']
    if skip_upload:
        cmd.append('--skip-upload')
    if local_phase4:
        cmd.append('--local-phase4')

    t = threading.Thread(target=_pipeline_thread, args=(cmd,), daemon=True)
    t.start()
    return jsonify({'ok': True, 'cmd': ' '.join(cmd)})


@app.route('/api/train/status', methods=['GET'])
def train_status():
    st = _train_state
    return jsonify({
        'status':  st['status'],
        'lastRun': st.get('last_run') or _read_last_run(),
    })


@app.route('/api/train/stream', methods=['GET'])
def train_stream():
    def generate():
        while True:
            try:
                item = _train_state['queue'].get(timeout=30)
                yield f"data: {json.dumps(item)}\n\n"
                if item.get('status') in ('done', 'failed', 'cancelled'):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'ping': 1})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/train/cancel', methods=['POST'])
def train_cancel():
    st = _train_state
    with st['lock']:
        proc = st.get('proc')
        if proc and proc.poll() is None:
            proc.terminate()
            st['status'] = 'cancelled'
            st['proc']   = None
            st['queue'].put({'status': 'cancelled'})
            return jsonify({'ok': True})
    return jsonify({'ok': False, 'error': 'No running pipeline'})


if __name__ == '__main__':
    load_models()
    app.run(host='127.0.0.1', port=5001, debug=False)
