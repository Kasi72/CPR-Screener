# NSE CPR Screener

**Intraday signal screener for NSE India equities — CPR × Camarilla Pivots × 4-Phase ML Pipeline**

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Node](https://img.shields.io/badge/node-18%2B-green.svg)](https://nodejs.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Overview

NSE CPR Screener identifies high-probability intraday setups across Nifty 500 stocks by combining classical pivot-based rule signals with a four-phase machine-learning stack. The system fires when 2+ CPR/Camarilla rules confluently trigger, then scores each signal with an ensemble of LightGBM, XGBoost, LSTM, and PPO models trained on 1.4 million historical signals.

**Live deployment:** Vercel (Node.js frontend) + Flask prediction microservice (Python backend)

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  NSE CPR Screener                   │
├──────────────┬──────────────────────────────────────┤
│  Frontend    │  server.js (Express/Node, Vercel)     │
│              │  public/index.html (vanilla JS)       │
│              │  ml_engine.js (HMM Viterbi, JS)       │
├──────────────┼──────────────────────────────────────┤
│  ML Backend  │  predict_server.py (Flask :5001)      │
│              │  └── /predict          XGBoost        │
│              │  └── /predict_ensemble Stacking       │
│              │  └── /regime           HMM state      │
│              │  └── /position_size    PPO sizing      │
│              │  └── /gate_weights     SHAP weights    │
├──────────────┼──────────────────────────────────────┤
│  ML Pipeline │  Phase 1: HMM + LightGBM              │
│              │  Phase 2: SHAP gates + Conformal       │
│              │  Phase 3: LSTM + Stacking meta-learner │
│              │  Phase 4: PPO position sizing          │
└──────────────┴──────────────────────────────────────┘
```

### Signal Generation (Rule Engine)

Eleven CPR/Camarilla rules fire when price interacts with pivot levels. A signal is accepted only when **≥ 2 rules confluently fire** (confluence gate). Each fired-rule combination generates one training row with 38 features and a `hit_t1` label (did price reach the +2.5% profit target within 5 bars?).

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

### Phase 2 — SHAP Gate Weights + Conformal Calibration

- Computes SHAP feature importances (LightGBM) and XGBoost gain importances
- Maps importances → rule gate weights (range 0.5–2.0) written to `models/shap_gate_weights.json`
- Calibrates conformal prediction sets on a 20% held-out split
- Re-trains an XGBoost classifier aligned with the current 38-feature schema

**Output:** `models/xgb_phase2.json`, `models/conformal_scores.json`, `models/shap_gate_weights.json`

### Phase 3 — LSTM + Stacking Ensemble

- LSTM (2-layer, 128 hidden) processes 20-bar sequence windows of `[ret, hl_range, vol_ratio, rsi14, sg_vel, mom5]`
- Logistic regression meta-learner stacks XGB + LGBM + LSTM predictions
- Trained with GPU acceleration (Kaggle T4 recommended; ~20 min vs 10 hr CPU)

**Output:** `models/lstm_model.pt`, `models/stacking_weights.json`, `models/meta_lgbm.txt`

### Phase 4 — PPO Position Sizing

- Proximal Policy Optimization (Stable-Baselines3) trained on a custom Gymnasium environment
- State: `[stack_score, regime_int, atr_pct, vol_rank, india_vix]`
- Action: position size 0–1 (continuous)
- Reward: risk-adjusted return clipped at ±3 ATR

**Output:** `models/ppo_policy_weights.json`

---

## Feature Engineering (38 Features)

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
| 21 | `pcr` | Options put-call ratio (stock or market Nifty fallback) |
| 22 | `india_vix` | India VIX (market fear gauge) |
| 23 | `conf_vol` | `n_rules_fired × vol_accel` (confluence × surge) |
| 24 | `rsi_dir` | `rsi14 × direction` (RSI alignment) |
| 25 | `hi52_dir` | `dist_hi52 × direction` (proximity × direction) |
| 26 | `cpr_compress` | Today CPR width / 5-day avg (squeeze detection) |
| 27 | `cpr_pos` | Close position within CPR [0 = lower, 1 = upper] |
| 28 | `dist_r1` | Distance of close from R1 pivot |
| 29 | `dist_s1` | Distance of close from S1 pivot |
| 30 | `mom3` | 3-day return |
| 31 | `mom10` | 10-day return |
| 32 | `mom20` | 20-day return |
| 33 | `rsi_div` | RSI divergence: +1 bullish, −1 bearish, 0 none |
| 34 | `vol_accel_delta` | Change in vol_accel vs prior day |
| 35 | `days_since_52hi` | Days since last 52-week high (momentum age) |
| 36 | `expiry_dist` | Calendar days to next monthly F&O expiry |
| 37 | `regime_stability` | `max_post_today − max_post_yesterday` (HMM confidence change) |
| 38 | `transition_risk` | `1 − max_posterior` (probability of regime ambiguity) |

Features 37–38 are joined from HMM posteriors at training time; at inference, neutral defaults (0.0 / 0.25) are used when live posteriors are unavailable.

---

## Repository Structure

```
nse-screener/
├── server.js                  # Express server (frontend + API proxy)
├── predict_server.py          # Flask ML inference server (port 5001)
├── ml_engine.js               # HMM Viterbi + feature scoring (browser/Node)
├── public/
│   └── index.html             # Single-page screener UI
├── models/                    # Trained model artifacts
│   ├── lgbm_model.txt         # Global LightGBM (38 features)
│   ├── lgbm_regime_{0-3}.txt  # Per-regime LightGBM sub-models
│   ├── lgbm_rule{1-11}.txt    # Per-rule LightGBM sub-models (after retrain)
│   ├── xgb_phase2.json        # XGBoost (Phase 2)
│   ├── hmm_params.json        # HMM matrices + regime map (JS Viterbi)
│   ├── hmm_posteriors.json    # Per-date posterior distributions
│   ├── stacking_weights.json  # Logistic meta-learner weights
│   ├── shap_gate_weights.json # Rule gate weights from SHAP
│   ├── conformal_scores.json  # Conformal calibration scores
│   └── ppo_policy_weights.json# PPO policy (position sizing)
├── scripts/
│   └── ml/
│       ├── data_utils.py      # FEATURE_COLS, TA helpers, feature API
│       ├── build_dataset.py   # Build signal_dataset.csv (38 features)
│       ├── train_phase1.py    # HMM + LGBM + regime/rule sub-models
│       ├── train_phase2.py    # SHAP gates + conformal calibration
│       ├── train_phase3.py    # LSTM + stacking ensemble
│       ├── train_phase4.py    # PPO position sizing
│       ├── run_all.py         # Full pipeline orchestrator
│       ├── scoring.py         # Inference helpers (stacking, conformal, PPO)
│       ├── lstm_model.py      # LSTMSignalModel (PyTorch)
│       ├── sector_features.py # Sector RS download + computation
│       ├── download_bhavcopy.py # NSE delivery % download
│       ├── download_pcr.py    # NSE FO bhavcopy PCR (stock + OPTIDX market)
│       └── sector_features.py # Sector index relative strength
├── requirements.txt           # Python dependencies (inference server)
├── package.json               # Node dependencies
└── vercel.json                # Vercel deployment config
```

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- NSE Nifty 500 OHLCV CSV at path configured in `data_utils.py` → `DATA_FILE`

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Node dependencies

```bash
npm install
```

### 3. Download supporting data (optional but recommended)

```bash
# NSE delivery % (smart money proxy, ~2-3 GB download)
python scripts/ml/download_bhavcopy.py

# NSE FO options PCR — stock options + Nifty market PCR
python scripts/ml/download_pcr.py

# Sector index closes (Nifty Bank, IT, Auto, Pharma, etc.)
python scripts/ml/sector_features.py
```

### 4. Build signal dataset

Requires the OHLCV CSV (~500 symbols × 5+ years):

```bash
python scripts/ml/build_dataset.py
```

Output: `models/signal_dataset.csv` (~400 MB, 1.4M signals, 38 features + labels)

### 5. Train the full ML pipeline

```bash
# Full pipeline (all 4 phases)
python scripts/ml/run_all.py

# Skip dataset rebuild (already built)
python scripts/ml/run_all.py --skip-dataset

# Skip individual phases
python scripts/ml/run_all.py --skip-phase1 --skip-phase2
```

Phase timings (CPU, Nifty 500 universe):

| Phase | Description | Approx. Time |
|-------|-------------|-------------|
| Dataset build | 1.4M signals, 38 features | 3–5 hr |
| Phase 1 | HMM + LGBM + 4 regime + 11 rule sub-models | 3–4 hr |
| Phase 2 | SHAP + conformal | 30 min |
| Phase 3 | LSTM (CPU) | 8–10 hr |
| Phase 3 | LSTM (Kaggle T4 GPU) | ~20 min |
| Phase 4 | PPO training | 20–30 min |

### 6. Start the servers

```bash
# ML inference server (port 5001)
python predict_server.py

# Frontend server (port 3000)
npm run dev
```

---

## API Reference

All endpoints are served by `predict_server.py` on port 5001.

### `POST /predict`

Single XGBoost score (backward-compatible).

```json
Request: { "cpr_width_pct": 0.3, "vol_rank": 2.1, "rule_id": 3, ... }
Response: { "predictions": [0.412] }
```

### `POST /predict_ensemble`

Full ensemble score: XGB + LGBM + regime routing + per-rule blend + stacking.

```json
Response: {
  "xgb_score":    0.38,
  "lgbm_score":   0.41,
  "regime_score": 0.44,
  "soft_score":   0.42,
  "stack_score":  0.45,
  "lo":           0.31,
  "hi":           0.58
}
```

### `GET /regime`

Current HMM market regime.

```json
{ "regime": "Bull-Trend", "state": 0, "posterior": [0.82, 0.05, 0.10, 0.03] }
```

### `POST /position_size`

PPO-recommended position size.

```json
Request: { "stack_score": 0.45, "atr_pct": 0.012, "vol_rank": 1.8, "india_vix": 14.2 }
Response: { "position_fraction": 0.72, "regime_score": 0.82 }
```

### `GET /gate_weights`

SHAP-derived rule gate weights for UI rendering.

```json
{ "volSurge": 1.84, "ema200": 1.62, "adx": 1.41, ... }
```

### `GET /health`

Liveness check.

---

## Deployment

### Vercel (frontend)

```bash
npx vercel --prod --yes
```

The frontend (`server.js`) proxies `/predict*` and `/regime` calls to the ML backend. Set `ML_SERVER_URL` environment variable in Vercel project settings.

### ML Backend

Run `predict_server.py` on any Linux VPS or cloud VM with Python 3.10+ and the `models/` directory present. The server loads all model files at startup and serves predictions with a threading lock.

---

## Retraining Schedule

The pipeline is designed for monthly refresh:

1. Run `python scripts/ml/download_bhavcopy.py` to extend delivery data
2. Run `python scripts/ml/download_pcr.py` to extend PCR data
3. Run `python scripts/ml/build_dataset.py` to rebuild the signal dataset
4. Run `python scripts/ml/run_all.py --skip-dataset` to retrain all phases

A Windows Task Scheduler task (`UC_XGB_AutoRetrain`) fires monthly to automate steps 3–4.

---

## Performance Metrics (v2.0.0, OOS holdout)

| Model | Test AUC | Notes |
|-------|----------|-------|
| LightGBM global | 0.624 | 38 features, Optuna HPO |
| XGBoost Phase 2 | 0.618 | 38 features |
| Regime soft-blend | 0.651 | Posterior-weighted 4-state blend |
| LSTM Phase 3 | 0.576 | 20-bar sequence, CPU run |
| Stack ensemble | ~0.678 | XGB + LGBM + LSTM + regime meta-learner |

Win rate at threshold 0.30: ~38–42% with 2.5% profit target, 0.8% hard stop (5-bar hold).

---

## Technical Notes

### Windows OOM Fix (signal_dataset.csv)

After HMM training, Windows C heap fragmentation prevents the LightGBM phase from allocating even 128 KiB contiguous blocks. The fix: Phase 1B (`train_lgbm`) runs in a **fresh subprocess** (`--lgbm-only` flag), so the OS reclaims all HMM heap pages before LightGBM starts. `memory_map=True` in pandas prevents the C parser's large contiguous buffer allocation.

### HMM Posterior Features

`regime_stability` and `transition_risk` are derived from `hmm_posteriors.json` and joined to the signal dataset at training time. At inference (predict_server), neutral defaults (0.0, 0.25) are used unless a live posterior lookup is wired up.

### PCR Coverage

Per-symbol stock options PCR (OPTSTK) has ~15–30% date coverage. When missing, the system falls back to market-wide Nifty OPTIDX PCR (extracted separately as `__MKT_NIFTY__` rows), achieving near-100% date coverage.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Run the linter: `python -m py_compile scripts/ml/*.py`
4. Commit with conventional commits: `feat:`, `fix:`, `perf:`, `docs:`
5. Open a pull request

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built for NSE India intraday trading research. Not financial advice.*
