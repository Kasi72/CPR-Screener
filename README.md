# NSE CPR Screener

**Intraday signal screener for NSE India equities — CPR × Camarilla Pivots × 4-Phase ML Pipeline**

[![Version](https://img.shields.io/badge/version-2.4.1-blue.svg)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Node](https://img.shields.io/badge/node-18%2B-green.svg)](https://nodejs.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Overview

NSE CPR Screener identifies high-probability intraday setups across Nifty 500 stocks by combining classical pivot-based rule signals with a four-phase machine-learning stack. The system fires when 2+ CPR/Camarilla rules confluently trigger, scores each signal with a regime-conditional LightGBM ensemble trained on 1.4 million historical signals, and recommends position sizes via a PPO reinforcement-learning agent.

**Live deployment:** Vercel (Node.js frontend) + Flask prediction microservice (Python backend)

**Current model:** Phase 2c LightGBM — 63 features, regime-conditional, test AUC 0.6112 (global), 0.6465 (recent data)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     NSE CPR Screener                        │
├──────────────┬──────────────────────────────────────────────┤
│  Frontend    │  server.js (Express/Node, Vercel)             │
│              │  public/index.html (vanilla JS)               │
│              │  ml_engine.js (HMM Viterbi, JS)               │
├──────────────┼──────────────────────────────────────────────┤
│  ML Backend  │  predict_server.py (Flask :5001)              │
│              │  └── /predict          XGBoost fallback        │
│              │  └── /predict_ensemble Phase 2c primary       │
│              │  └── /regime           HMM state               │
│              │  └── /position_size    PPO sizing              │
│              │  └── /gate_weights     SHAP weights            │
├──────────────┼──────────────────────────────────────────────┤
│  ML Pipeline │  Phase 1: HMM + LightGBM + regime sub-models │
│              │  Phase 2: SHAP gates + conformal calibration  │
│              │  Phase 2b: LightGBM HPO (Optuna 100-trial)    │
│              │  Phase 2c: Regime-conditional LGBM (active)   │
│              │  Phase 3: Meta-stacking (BYPASSED — hurts AUC)│
│              │  Phase 4: PPO position sizing                  │
└──────────────┴──────────────────────────────────────────────┘
```

### Why Phase 3 is bypassed

Comprehensive ablation testing showed meta-stacking (XGB + LGBM + regime + soft-blend stacked via a meta-LightGBM) reduces AUC vs the Phase 2c model used alone: stack AUC 0.6116 vs Phase 2c AUC 0.6932 on the full dataset (0.6465 recent). The meta-learner is trained and available but Phase 2c predictions are routed directly to the PPO sizer.

### Signal Generation (Rule Engine)

Eleven CPR/Camarilla rules fire when price interacts with pivot levels. A signal is accepted only when **≥ 2 rules confluently fire** (confluence gate). Each fired-rule combination generates one training row with up to 63 features and a `hit_t1` label (did price reach the +2.5% profit target within 5 bars?).

| Rule | Condition |
|------|-----------|
| rule1 | S3/R3 Camarilla level inside CPR (rare compression) |
| rule2 | CPR width < 0.5% (narrow pivot = breakout expected) |
| rule3 | Price crosses CPR upper band |
| rule4 | Previous day H/L entirely outside CPR |
| rule5 | VWAP inside CPR zone |
| rule6 | Wide CPR + price near S3 or R3 |
| rule7 | Price retesting CPR band from outside |
| rule8 | Price above CPR + prior day high above CPR |
| rule9 | Price >2% from pivot (extended, mean-reversion) |
| rule10 | Prior day H/L tested CPR band |
| rule11 | Price between VWAP and CPR band (squeeze) |

---

## 4-Phase ML Pipeline

### Phase 1 — HMM Regime Detection + LightGBM

**Phase 1A — HMM:** Trains a 4-state Gaussian HMM on 5 years of Nifty 50 daily observations (`ret`, `vol20`, `trend`, `vol_ratio`, `sg_vel`). States are mapped to named regimes by mean return rank:

| State | Regime | Win-rate bias |
|-------|--------|---------------|
| Highest mean ret | Bull-Trend | Longs outperform |
| 2nd | Bear-Trend | Shorts outperform |
| 3rd | Chop | Signals weaker |
| Lowest | High-Vol-Panic | Avoid |

**Phase 1B — LightGBM global model** with 50-trial Optuna HPO. Trained on all signals, temporal 80/20 split, monotone constraints enforced on directional features.

**Phase 1B+ — Regime sub-models:** 4 per-regime LightGBM models (`lgbm_regime_0..3.txt`) with 20-trial per-regime Optuna HPO. At inference, predictions are posterior-weighted soft-blended: `Σ P(state|obs) × P(win|x, state)`.

**Phase 1B++ — Per-rule sub-models:** 11 rule-specific LightGBM models (`lgbm_rule1..11.txt`) capturing rule-specific win patterns. Inference blends 70% global + 30% rule-specific.

**Outputs:** `models/lgbm_model.txt`, `models/lgbm_regime_{0-3}.txt`, `models/lgbm_rule{1-11}.txt`, `models/hmm_params.json`, `models/hmm_posteriors.json`

---

### Phase 2 — SHAP Gate Weights + Conformal Calibration

- Computes SHAP feature importances (LightGBM) and XGBoost gain importances
- Maps importances → rule gate weights (range 0.5–2.0) written to `models/shap_gate_weights.json`
- Calibrates conformal prediction sets on a 20% held-out split
- Re-trains an XGBoost classifier aligned with the 38-feature schema

**Output:** `models/xgb_phase2.json`, `models/conformal_scores.json`, `models/shap_gate_weights.json`

---

### Phase 2b — LightGBM HPO (Kaggle)

100-trial Optuna hyperparameter optimisation on 1.39M signals, 38-feature schema. Establishes the baseline HPO parameter set used in Phase 2c.

**Metrics:** CV AUC 0.5958, Val AUC 0.6124

**Output:** `models/lgbm2b_params.json`

---

### Phase 2c — Regime-Conditional LightGBM (Primary Scorer)

The active production scorer. Trains one global LightGBM + 4 per-regime LightGBM models on 63 features (56 base + 7 interaction features). Regime assignment gates which sub-model scores each signal; global model provides fallback.

**63-feature schema:**
- 36 core signal features (CPR, VWAP, ATR, momentum, RSI, market/sector RS, delivery %, PCR, India VIX)
- 5 Sprint 1 CPR depth features (`cpr_overlap_pct`, `open_to_cpr_dist`, `prev_cpr_respected`, `cpr_zone_vol_ratio`, `hmm_regime`)
- 5 Sprint 2A CPR structure features (`open_inside_cpr`, `cpr_virgin`, `consecutive_narrow_cprs`, `cpr_midpoint_trend`, `cpr_expansion_factor`)
- 5 Sprint 2B CPR context features (`cpr_above_prev_cpr`, `prev_close_inside_cpr`, `atr_to_cpr_ratio`, `cpr_width_percentile_252d`, `prev_day_ochoa_type`)
- 5 Sprint 3 gap/bar features (`gap_pct`, `cpr_test_count_5d`, `prev_bar_close_pos`, `atr_expansion`, `vol_trend_slope`)
- 7 interaction features (`cpr_vol_interaction`, `regime_momentum`, `cpr_rsi_squeeze`, `overlap_vol_signal`, `rs_direction_alignment`, `virgin_momentum`, `narrow_breakout_vol`)

**Metrics:** Global test AUC 0.6112, recent-data AUC 0.6465, full-dataset AUC 0.6932; runtime 193 min (Kaggle T4)

**Output:** `models/lgbm2c_global.txt`, `models/lgbm2c_regime_{0-3}.txt`, `models/shap_weights2c.json`

---

### Phase 3 — LSTM + Stacking Ensemble (Available, Bypassed)

- LSTM (2-layer, 128 hidden) processes 20-bar sequence windows
- Logistic regression meta-learner stacks XGB + LGBM + regime + soft + Phase 2c predictions
- **Bypassed in production**: Phase 2c alone outperforms the stack (AUC 0.6932 vs 0.6116)

**Output:** `models/lstm_model.pt` (optional), `models/stacking_weights.json`, `models/meta_lgbm.txt`

---

### Phase 4 — PPO Position Sizing

Proximal Policy Optimization (Stable-Baselines3) trained on a Gymnasium environment where each step is one historical signal.

**State (60-dim):** 56 base signal features + `[lgbm2c_score, exposure, cumulative_pnl, win_streak]`

**Action:** Discrete(5) → position size ∈ {0%, 25%, 50%, 75%, 100%}

**Reward:** `trade_return × size_frac − 0.001 × |action_delta|` (P&L minus transaction cost)

**Training:** 200k timesteps, early-stop after 8 evaluations without improvement (eval every 5k steps)

**Metrics (test set, 209k signals):**
- Sharpe (full size): −3.337
- Sharpe (PPO-sized): −3.375
- Average position size: 72.5%

**Output:** `models/ppo_policy.zip`, `models/ppo_policy_weights.json`

---

## Feature Engineering

### Base Features (43, live inference)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `cpr_width_pct` | CPR width as % of pivot |
| 2 | `vwap_dist` | Distance of close from VWAP |
| 3 | `atr_pct_rank` | 252-day percentile rank of current ATR |
| 4 | `vol_rank` | Volume / 20-day avg volume |
| 5 | `n_rules_fired` | Number of confluent rules |
| 6 | `sg_vel` | Savitzky-Golay velocity (price momentum) |
| 7 | `ema200_dist` | Distance from 200-day EMA |
| 8 | `rsi14` | Wilder RSI (14-period) |
| 9 | `mom5` | 5-day return |
| 10 | `dow` | Day of week (0=Mon) |
| 11 | `rule_id` | Triggering rule integer (1–11) |
| 12 | `direction` | Signal direction (+1 long / −1 short) |
| 13 | `dist_hi52` | Distance below 52-week high (≤ 0) |
| 14 | `dist_lo52` | Distance above 52-week low (≥ 0) |
| 15 | `vol_accel` | Volume today / 5-day avg (surge ratio) |
| 16 | `market_rs_5d` | Stock / Nifty 5-day relative strength |
| 17 | `market_rs_20d` | Stock / Nifty 20-day relative strength |
| 18 | `sector_rs_5d` | Stock / sector index 5-day RS |
| 19 | `sector_rs_20d` | Stock / sector index 20-day RS |
| 20 | `deliv_pct` | NSE delivery % (smart-money proxy) |
| 21 | `pcr` | Options put-call ratio |
| 22 | `india_vix` | India VIX (market fear gauge) |
| 23 | `conf_vol` | `n_rules_fired × vol_accel` |
| 24 | `rsi_dir` | `rsi14 × direction` |
| 25 | `hi52_dir` | `dist_hi52 × direction` |
| 26 | `cpr_compress` | Today CPR width / 5-day avg (squeeze) |
| 27 | `cpr_pos` | Close position within CPR band [0–1] |
| 28 | `dist_r1` | Distance of close from R1 pivot |
| 29 | `dist_s1` | Distance of close from S1 pivot |
| 30 | `mom3` | 3-day return |
| 31 | `mom10` | 10-day return |
| 32 | `mom20` | 20-day return |
| 33 | `rsi_div` | RSI divergence (+1 bull / −1 bear / 0) |
| 34 | `vol_accel_delta` | Change in vol surge ratio vs prior day |
| 35 | `days_since_52hi` | Days since last 52-week high |
| 36 | `expiry_dist` | Days to next monthly F&O expiry |
| 37 | `regime_stability` | HMM max posterior delta (day-over-day) |
| 38 | `transition_risk` | `1 − max_posterior` (regime ambiguity) |
| 39 | `gap_pct` | Open gap vs prior close (%) |
| 40 | `cpr_test_count_5d` | Times price tested CPR in past 5 sessions |
| 41 | `prev_bar_close_pos` | Prior bar close position relative to CPR |
| 42 | `atr_expansion` | ATR today / ATR 10-day avg |
| 43 | `vol_trend_slope` | 20-day volume slope (declining/rising) |

### Additional CPR Depth Features (Phase 2c training, 20 features)

Sprint 1–2B CPR features used in Kaggle training only (see `kaggle/phase2c/` for exact extraction logic):

- **Sprint 1:** `cpr_overlap_pct`, `open_to_cpr_dist`, `prev_cpr_respected`, `cpr_zone_vol_ratio`, `hmm_regime`
- **Sprint 2A:** `open_inside_cpr`, `cpr_virgin`, `consecutive_narrow_cprs`, `cpr_midpoint_trend`, `cpr_expansion_factor`
- **Sprint 2B:** `cpr_above_prev_cpr`, `prev_close_inside_cpr`, `atr_to_cpr_ratio`, `cpr_width_percentile_252d`, `prev_day_ochoa_type`

### Interaction Features (Phase 2c training, 7 features)

Computed from base + CPR depth features: `cpr_vol_interaction`, `regime_momentum`, `cpr_rsi_squeeze`, `overlap_vol_signal`, `rs_direction_alignment`, `virgin_momentum`, `narrow_breakout_vol`

---

## Repository Structure

```
nse-screener/
├── server.js                      # Express server (frontend + API proxy)
├── predict_server.py              # Flask ML inference server (port 5001)
├── ml_engine.js                   # HMM Viterbi + feature scoring (browser/Node)
├── public/
│   └── index.html                 # Single-page screener UI
├── models/                        # Trained model artifacts
│   ├── lgbm_model.txt             # Global LightGBM Phase 1 (38 features)
│   ├── lgbm_regime_{0-3}.txt      # Per-regime LightGBM sub-models (Phase 1)
│   ├── lgbm_rule{1-11}.txt        # Per-rule LightGBM sub-models
│   ├── lgbm2c_global.txt          # Phase 2c global model (63 features)
│   ├── lgbm2c_regime_{0-3}.txt    # Phase 2c per-regime models (63 features)
│   ├── xgb_phase2.json            # XGBoost Phase 2 (38 features)
│   ├── hmm_params.json            # HMM matrices + regime map
│   ├── hmm_posteriors.json        # Per-date posterior distributions
│   ├── stacking_weights.json      # Logistic meta-learner weights (Phase 3)
│   ├── meta_lgbm.txt              # Phase 3 meta-stacker (BYPASSED)
│   ├── shap_gate_weights.json     # Rule gate weights from SHAP
│   ├── shap_weights2c.json        # Phase 2c SHAP importances
│   ├── conformal_scores.json      # Conformal calibration scores
│   ├── ppo_policy.zip             # PPO policy (SB3 format)
│   ├── ppo_policy_weights.json    # PPO policy weights (JSON, lightweight)
│   ├── phase{1..4}_metrics.json   # Per-phase training metrics
│   └── signal_dataset.csv         # ⚠ gitignored — ~400MB, 1.4M signals
├── kaggle/
│   ├── phase2c/
│   │   └── cpr_phase2c_kernel.py  # Phase 2c Kaggle kernel (63-feat LGBM)
│   └── phase4/
│       └── cpr_phase4_kernel.py   # Phase 4 Kaggle kernel (PPO)
├── scripts/
│   └── ml/
│       ├── data_utils.py          # FEATURE_COLS (43), TA helpers, feature API
│       ├── build_dataset.py       # Build signal_dataset.csv (43 features)
│       ├── train_phase1.py        # HMM + LGBM + regime/rule sub-models
│       ├── train_phase2.py        # SHAP gates + conformal calibration
│       ├── train_phase3.py        # LSTM + stacking ensemble
│       ├── train_phase4.py        # PPO position sizing (local)
│       ├── run_all.py             # Full pipeline orchestrator
│       ├── scoring.py             # Inference helpers
│       ├── lstm_model.py          # LSTMSignalModel (PyTorch)
│       ├── kaggle_phase2b_runner.py # Kaggle Phase 2b HPO runner
│       ├── kaggle_phase3_runner.py  # Kaggle Phase 3 runner
│       ├── kaggle_phase4_runner.py  # Kaggle Phase 4 PPO runner
│       ├── post_phase4_deploy.py    # Post-training watcher + auto-deploy
│       ├── score_p2c.py             # Inject lgbm2c_score into dataset
│       ├── sector_features.py       # Sector RS download + computation
│       ├── download_bhavcopy.py     # NSE delivery % download
│       └── download_pcr.py          # NSE FO bhavcopy PCR download
├── requirements.txt               # Python dependencies
├── package.json                   # Node dependencies
└── vercel.json                    # Vercel deployment config
```

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- NSE Nifty 500 OHLCV CSV configured in `data_utils.py` → `DATA_FILE`
- Kaggle API credentials (`~/.kaggle/kaggle.json`) for GPU training phases

### 1. Install dependencies

```bash
pip install -r requirements.txt
npm install
```

### 2. Download supporting data (recommended)

```bash
# NSE delivery % (smart money proxy)
python scripts/ml/download_bhavcopy.py

# NSE FO options PCR (stock + Nifty market PCR)
python scripts/ml/download_pcr.py

# Sector index closes (Nifty Bank, IT, Auto, Pharma, etc.)
python scripts/ml/sector_features.py
```

### 3. Build signal dataset

```bash
python scripts/ml/build_dataset.py
```

Output: `models/signal_dataset.csv` (~400 MB, 1.4M signals, 43 features + labels)

### 4. Train the full ML pipeline

```bash
# Full pipeline (all 4 phases, local CPU)
python scripts/ml/run_all.py

# Skip dataset rebuild (already built)
python scripts/ml/run_all.py --skip-dataset

# Skip individual phases
python scripts/ml/run_all.py --skip-phase1 --skip-phase2
```

**Kaggle GPU training** (recommended for Phase 2c and Phase 4):

```bash
# Phase 2b: LightGBM HPO (Optuna 100 trials)
python scripts/ml/kaggle_phase2b_runner.py

# Phase 2c: Regime-conditional LGBM (63 features, Kaggle T4 ~3 hr)
# (See kaggle/phase2c/ — push kernel manually or adapt runner)

# Phase 4: PPO position sizing (Kaggle T4 ~20 min)
python scripts/ml/kaggle_phase4_runner.py
```

Phase timings (CPU unless noted):

| Phase | Description | Approx. Time |
|-------|-------------|-------------|
| Dataset build | 1.4M signals, 43 features | 3–5 hr |
| Phase 1 | HMM + LGBM + 4 regime + 11 rule sub-models | 3–4 hr |
| Phase 2 | SHAP + conformal | 30 min |
| Phase 2b | LightGBM HPO (Kaggle) | ~50 min |
| Phase 2c | Regime-conditional LGBM, 63 features (Kaggle T4) | ~3 hr |
| Phase 3 | LSTM + stacking (optional, bypassed) | 20 min (Kaggle) |
| Phase 4 | PPO training (Kaggle T4) | ~20 min |

### 5. Start the servers

```bash
# ML inference server (port 5001)
python predict_server.py

# Frontend server (port 3000)
npm run dev
```

---

## API Reference

All endpoints served by `predict_server.py` on port 5001.

### `POST /predict`

XGBoost score (backward-compatible fallback).

```json
Request: { "cpr_width_pct": 0.3, "vol_rank": 2.1, "rule_id": 3, ... }
Response: { "predictions": [0.412] }
```

### `POST /predict_ensemble`

Full ensemble score. Routes through Phase 2c (primary) with XGB/regime/soft as secondary signals.

```json
Response: {
  "xgb_score":     0.38,
  "lgbm_score":    0.41,
  "regime_score":  0.44,
  "soft_score":    0.42,
  "stack_score":   0.45,
  "lgbm2c_score":  0.51,
  "lo":            0.31,
  "hi":            0.58
}
```

`stack_score` routes to `lgbm2c_score` (Phase 2c) directly; the meta-stacker is bypassed.

### `GET /regime`

Current HMM market regime.

```json
{ "regime": "Bull-Trend", "state": 0, "posterior": [0.82, 0.05, 0.10, 0.03] }
```

### `POST /position_size`

PPO-recommended position fraction for a signal.

```json
Request: { "stack_score": 0.51, "atr_pct": 0.012, "vol_rank": 1.8, "india_vix": 14.2 }
Response: { "position_fraction": 0.72, "regime_score": 0.82 }
```

### `GET /gate_weights`

SHAP-derived rule gate weights for UI rendering.

```json
{ "volSurge": 1.84, "ema200": 1.62, "adx": 1.41, ... }
```

### `GET /health`

Liveness probe: `{ "status": "ok" }`

---

## Deployment

### Vercel (frontend)

```bash
npx vercel --prod --yes
```

The frontend (`server.js`) proxies `/predict*` and `/regime` calls to the ML backend. Set `ML_SERVER_URL` in Vercel project settings.

### ML Backend

Run `predict_server.py` on any Linux VPS with Python 3.10+ and the `models/` directory present. The server loads all model artifacts at startup with a threading lock on inference.

---

## Retraining Schedule

Monthly refresh recommended:

```bash
python scripts/ml/download_bhavcopy.py   # extend delivery data
python scripts/ml/download_pcr.py        # extend PCR data
python scripts/ml/build_dataset.py       # rebuild signal dataset
python scripts/ml/run_all.py --skip-dataset   # retrain all phases
```

A Windows Task Scheduler task (`UC_XGB_AutoRetrain`) fires monthly (next: 2026-09-16) to automate steps 3–4.

---

## Performance Metrics (v2.4.x, OOS holdout)

| Model | AUC | Notes |
|-------|-----|-------|
| LightGBM Phase 1 (global) | 0.571 | 38 features, Optuna HPO |
| Regime soft-blend | 0.644 | 4-state posterior-weighted blend |
| XGBoost Phase 2 | 0.619 | 38 features |
| LightGBM Phase 2b | 0.612 (val) | 100-trial Optuna HPO |
| **Phase 2c (global, test)** | **0.611** | **63 features — primary scorer** |
| Phase 2c (recent data) | 0.647 | Last 20% of temporal split |
| Phase 2c (full dataset) | 0.693 | All data |
| Phase 3 stack | 0.612 | XGB+LGBM+regime+soft+p2c — **bypassed** |
| Phase 4 PPO Sharpe | −3.375 | Sized vs −3.337 full-size |

Win rate at threshold 0.30: ~38–42% with 2.5% profit target, 0.8% hard stop (5-bar hold).

---

## Technical Notes

### Windows OOM Fix

After HMM training, Windows C heap fragmentation prevents LightGBM from allocating contiguous blocks. Fix: Phase 1B runs in a **fresh subprocess** (`--lgbm-only` flag), letting the OS reclaim all HMM heap pages. `memory_map=True` in pandas avoids the large contiguous buffer required by the C parser.

### Phase 2c vs data_utils.py Feature Schema

`data_utils.py` maintains 43 base features for live inference. Phase 2c was trained on Kaggle with 63 features (56 base including Sprint 1–2B CPR depth features + 7 interactions). At live inference, `predict_server.py` computes the interactions and passes 43 available base features; the Phase 2c model fills missing Sprint 1–2B CPR features with neutral defaults.

### PCR Coverage

Per-symbol stock options PCR (OPTSTK) covers ~15–30% of signal dates. Fallback: market-wide Nifty OPTIDX PCR stored as `__MKT_NIFTY__`, achieving near-100% date coverage.

### Phase 4 PPO Design Notes

- Sharpe-delta reward was tested (v2.4.0) but produced high variance and negative delta. Reverted to direct P&L reward in v2.4.1.
- `lgbm2c_score` is the primary quality signal in the PPO state — it encodes the ML estimate of signal quality directly, replacing the earlier `regime_score`.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Compile-check Python: `python -m py_compile scripts/ml/*.py`
4. Commit with conventional commits: `feat:`, `fix:`, `perf:`, `docs:`
5. Open a pull request

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built for NSE India intraday trading research. Not financial advice.*
