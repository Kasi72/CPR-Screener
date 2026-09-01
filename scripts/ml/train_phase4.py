"""
Phase 4: Proximal Policy Optimization (PPO) for Position Sizing

Run AFTER Phase 3:
    python scripts/ml/train_phase4.py

Outputs:
    models/ppo_policy.pt            — PPO Actor network weights (position sizer)
    models/phase4_metrics.json      — Training metrics

The PPO agent learns to size positions [0%, 25%, 50%, 75%, 100%] based on
signal quality, market regime, and portfolio state to maximise Sharpe ratio.
"""

import os, json, sys, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.ml.data_utils import MODELS_DIR, FEATURE_COLS, load_signal_dataset

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor


# ─────────────────────── Trading Environment ─────────────────────────────────

class SignalSizingEnv(gym.Env):
    """
    Custom Gym environment for signal position sizing.

    State  : [12 signal features + 1 regime_score + 1 current_exposure
              + 1 portfolio_sharpe + 1 win_streak]   → 16 dims
    Action : Discrete(5)  →  [0%, 25%, 50%, 75%, 100%]
    Reward : direction * actual_return * size_fraction
             - 0.001 * |action_change|   (transition cost)
             + 0.1 * (reward > 0)        (win bonus)
    """
    ACTION_SIZES  = np.array([0.0, 0.25, 0.50, 0.75, 1.00])
    STATE_DIM     = len(FEATURE_COLS) + 4   # 25 features + regime + exposure + sharpe + streak = 29

    def __init__(self, signals_df, mode='train'):
        super().__init__()
        self.df      = signals_df.reset_index(drop=True)
        self.mode    = mode
        self.n       = len(self.df)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.STATE_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.reset()

    def _get_obs(self):
        row = self.df.iloc[self.idx]
        base = np.array([float(row.get(f, 0)) for f in FEATURE_COLS], dtype=np.float32)
        extra = np.array([
            self.regime_score,
            self.exposure,
            np.clip(self.running_sharpe, -3, 3),
            float(self.win_streak),
        ], dtype=np.float32)
        obs = np.concatenate([base, extra])
        return np.clip(np.nan_to_num(obs), -10, 10).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.mode == 'train':
            self.idx = np.random.randint(0, max(1, self.n - 1))
        else:
            self.idx = 0
        self.exposure       = 0.0
        self.prev_action    = 0
        self.returns        = []
        self.running_sharpe = 0.0
        self.win_streak     = 0
        self.regime_score   = 0.5   # neutral; updated if HMM available
        return self._get_obs(), {}

    def step(self, action):
        row          = self.df.iloc[self.idx]
        size_frac    = self.ACTION_SIZES[action]
        actual_ret   = float(row.get('actual_return', 0.0))

        # Reward components
        trade_reward = actual_ret * size_frac                          # P&L contribution
        trans_cost   = 0.001 * abs(action - self.prev_action)         # friction
        win_bonus    = 0.05 if (actual_ret > 0.005 and size_frac > 0) else 0.0
        skip_penalty = 0.0 if size_frac > 0 else (0.01 if actual_ret > 0.01 else 0.0)
        reward       = trade_reward - trans_cost + win_bonus - skip_penalty

        # Update state
        self.returns.append(trade_reward)
        if len(self.returns) >= 10:
            r = np.array(self.returns[-30:])
            self.running_sharpe = (r.mean() / (r.std() + 1e-8)) * np.sqrt(252)
        self.win_streak = (self.win_streak + 1) if reward > 0 else 0
        self.exposure   = size_frac
        self.prev_action = action
        self.idx         = (self.idx + 1) % self.n

        done = (self.idx == 0)
        return self._get_obs(), float(reward), done, False, {
            'actual_return': actual_ret,
            'size': size_frac,
            'trade_reward': trade_reward,
        }


# ─────────────────────────── PPO Training ────────────────────────────────────

def train_ppo(df_train, df_eval):
    def make_env(df, mode):
        def _init():
            env = SignalSizingEnv(df, mode=mode)
            return Monitor(env)
        return _init

    train_env = DummyVecEnv([make_env(df_train, 'train')])
    eval_env  = DummyVecEnv([make_env(df_eval,  'eval')])

    policy_kwargs = dict(
        net_arch=[dict(pi=[128, 64], vf=[128, 64])],
        activation_fn=nn.ReLU
    )

    model = SB3PPO(
        'MlpPolicy',
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.90,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=42,
    )

    stop_cb  = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, verbose=1)
    eval_cb  = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_DIR,
        log_path=MODELS_DIR,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        callback_after_eval=stop_cb,
        verbose=1,
    )

    print("  Training PPO… (this may take 10-20 minutes)")
    model.learn(total_timesteps=50_000, callback=eval_cb)

    # Load best checkpoint
    best_path = os.path.join(MODELS_DIR, 'best_model')
    if os.path.exists(best_path + '.zip'):
        model = SB3PPO.load(best_path, env=train_env)
        print("  Loaded best checkpoint.")

    return model


def evaluate_ppo(model, df_test):
    """Simulate the PPO agent on test set and measure Sharpe improvement."""
    env = SignalSizingEnv(df_test, mode='eval')
    obs, _ = env.reset()

    ppo_returns, full_returns, sizes_taken = [], [], []
    for _ in range(len(df_test)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(int(action))
        ppo_returns.append(info['trade_reward'])
        full_returns.append(info['actual_return'])
        sizes_taken.append(info['size'])
        if done:
            break

    ppo_arr  = np.array(ppo_returns)
    full_arr = np.array(full_returns)

    sharpe_ppo  = ppo_arr.mean() / (ppo_arr.std() + 1e-8) * np.sqrt(252)
    sharpe_full = full_arr.mean() / (full_arr.std() + 1e-8) * np.sqrt(252)
    avg_size    = float(np.mean(sizes_taken)) if sizes_taken else 0.5

    print(f"\n  Sharpe (full size): {sharpe_full:.3f}")
    print(f"  Sharpe (PPO-sized): {sharpe_ppo:.3f}  ← improvement: {sharpe_ppo-sharpe_full:+.3f}")
    return {
        'sharpe_full':   round(float(sharpe_full), 4),
        'sharpe_ppo':    round(float(sharpe_ppo),  4),
        'sharpe_delta':  round(float(sharpe_ppo - sharpe_full), 4),
        'n_test':        len(ppo_returns),
    }


def export_policy_weights(model):
    """Extract policy network weights to plain dict for JS/ONNX inference."""
    policy = model.policy
    weights = {}
    for name, param in policy.named_parameters():
        if param.requires_grad:
            weights[name] = param.detach().cpu().numpy().tolist()
    return weights


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 60)
    print("  PHASE 4: PPO Position Sizer")
    print("=" * 60)

    df = load_signal_dataset()
    print(f"  {len(df)} signals loaded.")

    # Temporal split: 70% train, 15% eval, 15% test
    n      = len(df)
    n_tr   = int(n * 0.70)
    n_ev   = int(n * 0.15)
    df_tr  = df.iloc[:n_tr].copy()
    df_ev  = df.iloc[n_tr:n_tr+n_ev].copy()
    df_te  = df.iloc[n_tr+n_ev:].copy()

    print(f"  Train={len(df_tr)}  Eval={len(df_ev)}  Test={len(df_te)}")

    # Train PPO
    ppo_model = train_ppo(df_tr, df_ev)

    # Evaluate
    metrics = evaluate_ppo(ppo_model, df_te)

    # Export full PPO model (zip) + lightweight policy weights (json)
    out_pt = os.path.join(MODELS_DIR, 'ppo_policy')
    ppo_model.save(out_pt)
    print(f"\n  Saved PPO model → {out_pt}.zip")

    policy_weights = export_policy_weights(ppo_model)
    out_pw = os.path.join(MODELS_DIR, 'ppo_policy_weights.json')
    with open(out_pw, 'w') as f:
        json.dump(policy_weights, f)
    print(f"  Saved policy weights → {out_pw}")

    with open(os.path.join(MODELS_DIR, 'phase4_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("  Phase 4 complete. All phases done!")
    print("  Restart predict_server.py to load all new models.")
    print("=" * 60)


if __name__ == '__main__':
    main()
