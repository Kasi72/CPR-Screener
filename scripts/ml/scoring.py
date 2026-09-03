"""
scoring.py — Pure inference math shared by predict_server.py.

Functions here take model registry dict + raw values and return scores.
No Flask, no file I/O, no global state — all inputs explicit.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


ACTION_SIZES = [0.0, 0.25, 0.50, 0.75, 1.00]


def conformal_interval(
    raw_score: float,
    conformal_data: Optional[dict],
    alpha: float = 0.10,
) -> tuple[float, float]:
    """Return (lower, upper) conformal prediction interval."""
    if not conformal_data:
        return (max(0.0, raw_score - 0.15), min(1.0, raw_score + 0.15))
    scores = conformal_data.get('ensemble', [])
    if not scores:
        return (max(0.0, raw_score - 0.15), min(1.0, raw_score + 0.15))
    n = len(scores)
    q_idx = min(int(np.ceil((n + 1) * (1 - alpha))), n) - 1
    q = float(scores[max(0, q_idx)])
    return (max(0.0, raw_score - q), min(1.0, raw_score + q))


def stacking_ensemble(
    xgb_prob: float,
    lgbm_prob: float,
    meta_lgbm=None,
    stacking_weights: Optional[dict] = None,
    lstm_prob: Optional[float] = None,
    regime_hard: Optional[float] = None,
    soft_blend: Optional[float] = None,
    p2b_prob: Optional[float] = None,
    p2c_prob: Optional[float] = None,
) -> float:
    """
    Combine base model scores into a single ensemble probability.

    Priority:
      1. meta_lgbm (LightGBM meta-learner) if available
         - Sprint 4 retrained model expects 7 inputs: xgb/lgb/lstm/regime/soft/p2b/p2c
         - Pre-Sprint-4 model (5 inputs) still works via num_feature() check
      2. logistic regression fallback from stacking_weights.json
      3. simple average
    """
    if meta_lgbm is not None:
        lstm_v  = lstm_prob   if lstm_prob   is not None else (xgb_prob + lgbm_prob) / 2
        reg_h   = regime_hard if regime_hard is not None else lgbm_prob
        soft_v  = soft_blend  if soft_blend  is not None else reg_h
        p2b_v   = p2b_prob    if p2b_prob    is not None else (xgb_prob + lgbm_prob) / 2
        p2c_v   = p2c_prob    if p2c_prob    is not None else p2b_v

        n_feat = meta_lgbm.num_feature()
        if n_feat >= 7:
            meta_X = np.array([[xgb_prob, lgbm_prob, lstm_v, reg_h, soft_v, p2b_v, p2c_v]],
                               dtype=np.float32)
        else:
            # Pre-Sprint-4 model (5 inputs) — backward compat
            meta_X = np.array([[xgb_prob, lgbm_prob, lstm_v, reg_h, soft_v]], dtype=np.float32)
        return float(meta_lgbm.predict(meta_X)[0])

    sw = stacking_weights
    if not sw:
        return (xgb_prob + lgbm_prob) / 2

    w_xgb    = sw.get('xgb_weight',    1.0)
    w_lgbm   = sw.get('lgbm_weight',   1.0)
    w_lstm   = sw.get('lstm_weight',   0.0)
    w_regime = sw.get('regime_weight', 0.0)
    bias     = sw.get('intercept',     0.0)
    lstm_v   = lstm_prob if lstm_prob is not None else (xgb_prob + lgbm_prob) / 2
    reg_h    = regime_hard if regime_hard is not None else lgbm_prob
    log_odds = (w_xgb * xgb_prob + w_lgbm * lgbm_prob
                + w_lstm * lstm_v + w_regime * reg_h + bias)
    return float(1.0 / (1.0 + np.exp(-log_odds)))


def ppo_position_size(
    features_row: np.ndarray,
    ppo_weights: Optional[dict],
    regime_score: float = 0.5,
) -> float:
    """
    Map a feature row to a discrete position size in ACTION_SIZES.

    Uses exported PPO policy first layer as a linear approximation.
    Falls back to n_rules_fired heuristic if weights unavailable.
    """
    if not ppo_weights:
        score = float(features_row[4]) / 10.0   # n_rules_fired index
        score = float(np.clip(score, 0, 1)) * regime_score
        return ACTION_SIZES[min(int(score * 4), 4)]

    W0_data = ppo_weights.get('mlp_extractor.policy_net.0.weight')
    if W0_data is None:
        return ACTION_SIZES[2]  # default 50%

    W0 = np.array(W0_data, dtype=np.float32)          # [128, input_dim]
    x  = np.zeros(W0.shape[1], dtype=np.float32)
    n_feat = min(len(features_row), W0.shape[1] - 4)
    x[:n_feat] = features_row[:n_feat]
    if W0.shape[1] >= 4:
        x[-4] = regime_score  # regime at index -4 matches PPO state layout
    b0 = np.array(ppo_weights.get('mlp_extractor.policy_net.0.bias',
                                   np.zeros(W0.shape[0])), dtype=np.float32)
    h = np.maximum(0, W0 @ x + b0)

    W2_data = ppo_weights.get('mlp_extractor.policy_net.2.weight')
    if W2_data is not None:
        W2 = np.array(W2_data, dtype=np.float32)
        b2 = np.array(ppo_weights.get('mlp_extractor.policy_net.2.bias',
                                       np.zeros(W2.shape[0])), dtype=np.float32)
        h = np.maximum(0, W2 @ h + b2)

    Wa_data = ppo_weights.get('action_net.weight')
    if Wa_data is not None:
        Wa = np.array(Wa_data, dtype=np.float32)
        ba = np.array(ppo_weights.get('action_net.bias',
                                       np.zeros(Wa.shape[0])), dtype=np.float32)
        logits = Wa @ h + ba
        action = int(np.argmax(logits))
    else:
        action = int(np.argmax(h[:5]))  # fallback if action_net not exported
    return ACTION_SIZES[min(action, len(ACTION_SIZES) - 1)]
