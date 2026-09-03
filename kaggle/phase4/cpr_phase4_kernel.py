"""
cpr_phase4_kernel.py — Self-contained Phase 4 (PPO Position Sizing) for Kaggle.

Reads from /kaggle/input/cpr-screener-phase3-inputs/:
  signal_dataset.csv

Writes to /kaggle/working/:
  ppo_policy.zip           — SB3 PPO model (best checkpoint)
  ppo_policy_weights.json  — policy network weights as plain JSON
  phase4_metrics.json      — Sharpe improvement metrics
"""

import os, json, subprocess, sys, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# gymnasium preferred; fall back install for older Kaggle images
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gymnasium', 'shimmy'], check=False)
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except ImportError:
        import gym
        from gym import spaces

try:
    from stable_baselines3 import PPO as SB3PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'stable-baselines3[extra]'], check=False)
    from stable_baselines3 import PPO as SB3PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
    from stable_baselines3.common.monitor import Monitor

warnings.filterwarnings('ignore')

INPUT = '/kaggle/input/cpr-screener-phase4-inputs'
WORK  = '/kaggle/working'

# ── Feature columns (must match data_utils.py FEATURE_COLS exactly) ────────────
FEATURE_COLS = [
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
    'hmm_regime',
    # Sprint 2A
    'open_inside_cpr', 'cpr_virgin', 'consecutive_narrow_cprs',
    'cpr_midpoint_trend', 'cpr_expansion_factor',
    # Sprint 2B
    'cpr_above_prev_cpr', 'prev_close_inside_cpr', 'atr_to_cpr_ratio',
    'cpr_width_percentile_252d', 'prev_day_ochoa_type',
    # Sprint 3
    'gap_pct', 'cpr_test_count_5d', 'prev_bar_close_pos',
    'atr_expansion', 'vol_trend_slope',
]  # 56 base features

# State: 56 signal features + lgbm2c_score + exposure + running_sharpe + win_streak
STATE_DIM = len(FEATURE_COLS) + 4  # 56 + 4 = 60


# ── Trading Environment ────────────────────────────────────────────────────────

class SignalSizingEnv(gym.Env):
    """
    Gym environment for signal position sizing.
    State  : 56 signal features + [lgbm2c_score, exposure, running_sharpe, win_streak]
    Action : Discrete(5) → [0%, 25%, 50%, 75%, 100%]
    Reward : delta-Sharpe (change in rolling Sharpe) — directly optimizes what matters
    """
    ACTION_SIZES  = np.array([0.0, 0.25, 0.50, 0.75, 1.00])
    SHARPE_WINDOW = 50   # rolling window for reward computation

    def __init__(self, signals_df, mode='train'):
        super().__init__()
        self.df   = signals_df.reset_index(drop=True)
        self.mode = mode
        self.n    = len(self.df)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(STATE_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.reset()

    def _rolling_sharpe(self):
        if len(self.returns) < 10:
            return 0.0
        r = np.array(self.returns[-self.SHARPE_WINDOW:])
        return float((r.mean() / (r.std() + 1e-8)) * np.sqrt(252))

    def _get_obs(self):
        row  = self.df.iloc[self.idx]
        base = np.array([float(row.get(f, 0)) for f in FEATURE_COLS], dtype=np.float32)
        extra = np.array([
            float(row.get('lgbm2c_score', 0.5)),   # ML confidence — primary signal quality
            self.exposure,
            np.clip(self.running_sharpe, -3, 3),
            float(self.win_streak),
        ], dtype=np.float32)
        obs = np.concatenate([base, extra])
        return np.clip(np.nan_to_num(obs), -10, 10).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx            = np.random.randint(0, max(1, self.n - 1)) if self.mode == 'train' else 0
        self.exposure       = 0.0
        self.prev_action    = 0
        self.returns        = []
        self.running_sharpe = 0.0
        self.win_streak     = 0
        return self._get_obs(), {}

    def step(self, action):
        row        = self.df.iloc[self.idx]
        size_frac  = self.ACTION_SIZES[action]
        actual_ret = float(row.get('actual_return', 0.0))

        trade_return = actual_ret * size_frac

        # Sharpe-delta reward: directly maximise rolling Sharpe improvement
        prev_sharpe = self.running_sharpe
        self.returns.append(trade_return)
        new_sharpe  = self._rolling_sharpe()
        self.running_sharpe = new_sharpe

        sharpe_delta = new_sharpe - prev_sharpe
        trans_cost   = 0.001 * abs(action - self.prev_action)
        reward       = sharpe_delta - trans_cost

        self.win_streak  = (self.win_streak + 1) if trade_return > 0 and size_frac > 0 else 0
        self.exposure    = size_frac
        self.prev_action = action
        self.idx         = (self.idx + 1) % self.n
        done = (self.idx == 0)

        return self._get_obs(), float(reward), done, False, {
            'actual_return': actual_ret,
            'size': size_frac,
            'trade_reward': trade_return,
        }


# ── Load data ──────────────────────────────────────────────────────────────────

def load_data():
    import glob as _glob
    csv_path = os.path.join(INPUT, 'signal_dataset.csv')
    if not os.path.exists(csv_path):
        hits = _glob.glob(os.path.join(INPUT, '**', 'signal_dataset.csv'), recursive=True)
        if hits:
            csv_path = hits[0]
            print(f"  (found at {csv_path})")
        else:
            raise FileNotFoundError(
                f"signal_dataset.csv not found under {INPUT}. "
                f"Contents: {os.listdir(INPUT)}"
            )
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} signals loaded.")
    # Fill missing feature cols with 0 (lgbm2c_score default 0.5 = neutral)
    for col in FEATURE_COLS + ['actual_return', 'lgbm2c_score']:
        if col not in df.columns:
            df[col] = 0.5 if col == 'lgbm2c_score' else 0.0
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    df['lgbm2c_score'] = df['lgbm2c_score'].fillna(0.5)

    # Encode any string columns to numeric
    for col in FEATURE_COLS + ['actual_return', 'lgbm2c_score']:
        if col in df.columns and df[col].dtype == object:
            # rule_id: 'rule8' → 8, 'rule1' → 1, etc.
            # direction: 'BUY'/'SELL' → ordinal via category codes
            extracted = df[col].astype(str).str.extract(r'(\d+)')[0]
            if extracted.notna().mean() > 0.5:
                df[col] = pd.to_numeric(extracted, errors='coerce').fillna(0)
            else:
                df[col] = df[col].astype('category').cat.codes.astype(float)

    return df


# ── PPO Training ───────────────────────────────────────────────────────────────

def train_ppo(df_train, df_eval):
    def make_env(df, mode):
        def _init():
            return Monitor(SignalSizingEnv(df, mode=mode))
        return _init

    train_env = DummyVecEnv([make_env(df_train, 'train')])
    eval_env  = DummyVecEnv([make_env(df_eval,  'eval')])

    model = SB3PPO(
        'MlpPolicy',
        train_env,
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 64], vf=[128, 64])],
            activation_fn=nn.ReLU,
        ),
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

    stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=8, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=WORK,
        log_path=WORK,
        eval_freq=5_000,
        n_eval_episodes=1,
        deterministic=True,
        callback_after_eval=stop_cb,
        verbose=1,
    )

    print("  Training PPO (200k timesteps) ...")
    model.learn(total_timesteps=200_000, callback=eval_cb)

    best_path = os.path.join(WORK, 'best_model')
    if os.path.exists(best_path + '.zip'):
        model = SB3PPO.load(best_path, env=train_env)
        print("  Loaded best checkpoint.")

    return model


# ── Evaluate ───────────────────────────────────────────────────────────────────

def evaluate_ppo(model, df_test):
    env = SignalSizingEnv(df_test, mode='eval')
    obs, _ = env.reset()

    ppo_returns, full_returns, sizes = [], [], []
    for _ in range(len(df_test)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(int(action))
        ppo_returns.append(info['trade_reward'])
        full_returns.append(info['actual_return'])
        sizes.append(info['size'])
        if done:
            break

    ppo_arr  = np.array(ppo_returns)
    full_arr = np.array(full_returns)
    sharpe_ppo  = ppo_arr.mean()  / (ppo_arr.std()  + 1e-8) * np.sqrt(252)
    sharpe_full = full_arr.mean() / (full_arr.std()  + 1e-8) * np.sqrt(252)
    avg_size    = float(np.mean(sizes)) if sizes else 0.5

    print(f"\n  Sharpe (full size): {sharpe_full:.3f}")
    print(f"  Sharpe (PPO-sized): {sharpe_ppo:.3f}  (delta: {sharpe_ppo-sharpe_full:+.3f})")
    print(f"  Avg position size:  {avg_size:.2f}")
    return {
        'sharpe_full':  round(float(sharpe_full), 4),
        'sharpe_ppo':   round(float(sharpe_ppo),  4),
        'sharpe_delta': round(float(sharpe_ppo - sharpe_full), 4),
        'avg_size':     round(avg_size, 4),
        'n_test':       len(ppo_returns),
    }


# ── Export policy weights ──────────────────────────────────────────────────────

def export_policy_weights(model):
    weights = {}
    for name, param in model.policy.named_parameters():
        if param.requires_grad:
            weights[name] = param.detach().cpu().numpy().tolist()
    return weights


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CPR Phase 4: PPO Position Sizer")
    print("=" * 60)

    df = load_data()

    # Temporal split: 70/15/15
    n    = len(df)
    n_tr = int(n * 0.70)
    n_ev = int(n * 0.15)
    df_tr = df.iloc[:n_tr].copy()
    df_ev = df.iloc[n_tr:n_tr + n_ev].copy()
    df_te = df.iloc[n_tr + n_ev:].copy()
    print(f"  Train={len(df_tr):,}  Eval={len(df_ev):,}  Test={len(df_te):,}")

    model   = train_ppo(df_tr, df_ev)
    metrics = evaluate_ppo(model, df_te)

    # Save PPO model zip
    out_policy = os.path.join(WORK, 'ppo_policy')
    model.save(out_policy)
    print(f"\n  Saved → {out_policy}.zip")

    # Save lightweight policy weights JSON
    out_weights = os.path.join(WORK, 'ppo_policy_weights.json')
    with open(out_weights, 'w') as f:
        json.dump(export_policy_weights(model), f)
    print(f"  Saved → {out_weights}")

    # Save metrics
    out_metrics = os.path.join(WORK, 'phase4_metrics.json')
    with open(out_metrics, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved → {out_metrics}")

    print("\n" + "=" * 60)
    print("  Phase 4 complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
