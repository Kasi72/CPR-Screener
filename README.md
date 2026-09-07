# NSE CPR Screener

**Intraday signal screener for NSE India equities — CPR × Camarilla Pivots × 4-Phase ML Pipeline**

[![Version](https://img.shields.io/badge/version-2.5.0-blue.svg)](CHANGELOG.md)
[![Node](https://img.shields.io/badge/node-18%2B-green.svg)](https://nodejs.org/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Live](https://img.shields.io/badge/live-Vercel-black.svg)](https://cpr-screener.vercel.app)

> **© 2024–2026 Kasi (GitHub: Kasi72). All Rights Reserved.**
> Source-available for reference only — see [LICENSE](LICENSE).

---

## Overview

NSE CPR Screener identifies high-probability intraday setups across Nifty 500+ stocks by combining classical pivot-based rule signals with a four-phase machine-learning stack. The system fires when 2+ CPR/Camarilla rules confluently trigger, scores each signal with a regime-conditional LightGBM ensemble, and recommends position sizing via a PPO reinforcement-learning agent.

**v2.5 ships pure-JS inference** — the LGBM2c model and PPO position sizer run entirely in Node.js with no Python Flask dependency. The screener deploys as a single Vercel serverless app.

**Live deployment:** Vercel (Node.js — no Python backend required as of v2.5)

**Current model:** LGBM2c — 63 features, regime-conditional, India VIX as top feature (4,429 tree splits), test AUC 0.6112 (global), 0.6465 (recent data)

---

## What's New in v2.5

| Feature | Detail |
|---------|--------|
| **Pure-JS LGBM2c + PPO inference** | `lib/lgbmInfer.js` — no Python / Flask required |
| **India VIX integration** | Live VIX fetched via Yahoo Finance; top LGBM2c feature |
| **3-mode hold duration** | 3-day / 1-week / 3-week, each with MFE/MAE-calibrated targets & stops |
| **R:R engine overhaul** | Calibrated from 4.56M-signal MFE/MAE backtest on 1,414 NSE stocks |
| **High Yield filter** | rule4 + rule11, conf≥62, Bull-Trend regime, ≥2 rules — highest-precision subset |
| **Dynamic UC score weights** | `ucScoreWeights.ts` updated from backtest analysis |
| **Nifty regime gate** | Bull-Trend required for STRONG BUY (no signals in High-Vol-Panic) |
| **Circuit-day CPR fix** | Degenerate CPR (H=L) skipped in prevDay lookup |
| **Division-by-zero guards** | change%, live price refresh, LGBM2c returns, HMM covariance |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  NSE CPR Screener v2.5                      │
├──────────────┬──────────────────────────────────────────────┤
│  Frontend    │  public/index.html (vanilla JS)              │
│              │  ├─ CPR / Camarilla rule engine (rule1–11)   │
│              │  ├─ R:R engine (MFE/MAE calibrated, 3-mode)  │
│              │  ├─ High Yield filter                        │
│              │  └─ Live price refresh + tier display        │
├──────────────┼──────────────────────────────────────────────┤
│  Server      │  server.js (Express/Node, Vercel SSE)        │
│              │  ├─ /api/screen/stream  SSE scan endpoint    │
│              │  ├─ HMM Viterbi regime detection             │
│              │  ├─ LGBM2c inference (lib/lgbmInfer.js)      │
│              │  └─ PPO position sizing (pure JS)            │
├──────────────┼──────────────────────────────────────────────┤
│  ML Stack    │  Phase 1:  HMM regime + LightGBM global      │
│              │  Phase 2c: Regime-conditional LGBM (active)  │
│              │  Phase 4:  PPO position sizing (pure JS)     │
│              │  [Phase 3 meta-stack available, bypassed]    │
└──────────────┴──────────────────────────────────────────────┘
```

### Signal Generation (Rule Engine)

Eleven CPR/Camarilla rules fire when price interacts with pivot levels. Signal accepted only when **≥ 2 rules confluently fire**.

| Rule | Condition |
|------|-----------|
| rule1 | S3/R3 Camarilla level inside CPR (rare compression) |
| rule2 | CPR width < 0.5% (narrow pivot — breakout expected) |
| rule3 | Price crosses CPR upper band |
| rule4 | Previous day H/L entirely outside CPR |
| rule5 | VWAP inside CPR zone |
| rule6 | Wide CPR + price near S3 or R3 |
| rule7 | Price retesting CPR band from outside |
| rule8 | Price above CPR + prior day high above CPR |
| rule9 | Price >2% from pivot (extended, mean-reversion) |
| rule10 | Prior day H/L tested CPR band |
| rule11 | Price between VWAP and CPR band (squeeze) |

### Tier System

| Tier | Conditions |
|------|-----------|
| **STRONG BUY** | conf≥62 + R:R≥threshold + confluence + Bull-Trend regime |
| **BUY** | conf≥55 + R:R≥threshold + confluence |
| **SPECULATIVE** | conf≥45 + R:R≥SPEC threshold |
| **WATCH** | conf < threshold |
| **AVOID** | High-Vol-Panic regime or fails all gates |

---

## R:R Calibration (3-Mode Hold)

Targets and stops derived from 4.56M-signal MFE/MAE backtest across 1,414 NSE stocks.

| Mode | Hold | T1 target | T2 target | Stop | SPEC R:R | BUY R:R | STRONG R:R |
|------|------|-----------|-----------|------|----------|---------|------------|
| **3-day** (short) | 3 bars | ~2.5% | ~5% | ~3% | ≥1.4 | ≥1.5 | ≥1.6 |
| **1-week** (medium) | 5 bars | ~3.5% | ~7% | ~4% | ≥1.5 | ≥1.55 | ≥1.65 |
| **3-week** (swing) | 15 bars | ~6.5% | ~12% | ~7% | ≥1.592 | ≥1.624 | ≥1.728 |

R:R uses T2 (full target) not T1, making thresholds achievable from real MFE distributions.

---

## High Yield Filter

Derived from backtest findings: signals meeting all 5 conditions achieve the highest precision.

1. `rule4` or `rule11` fired (highest-precision rules from backtest)
2. `conf ≥ 62` (LGBM2c confidence)
3. `regime === 'Bull-Trend'` (HMM state)
4. `matchCount ≥ 2` (confluence gate)

Expected EV at these conditions: **+1.98%** per signal (backtest, LGBM2c + India VIX scoring).

---

## 4-Phase ML Pipeline

### Phase 1 — HMM Regime Detection + LightGBM

**HMM:** 4-state Gaussian HMM on 5 years of Nifty 50 daily observations. States mapped to named regimes by mean return rank.

| State | Regime | Trading bias |
|-------|--------|--------------|
| Highest mean ret | Bull-Trend | Longs outperform |
| 2nd | Bear-Trend | Shorts outperform |
| 3rd | Chop | Signals weaker |
| Lowest | High-Vol-Panic | Avoid entirely |

**LightGBM global:** 38 features, 50-trial Optuna HPO, temporal 80/20 split.

**Regime sub-models:** 4 per-regime LightGBM models, posterior-weighted soft-blend at inference.

**Rule sub-models:** 11 rule-specific models, blended 70% global + 30% rule-specific.

---

### Phase 2c — Regime-Conditional LightGBM (Primary Scorer)

Active production scorer. 63 features, regime-gated inference, India VIX as top feature.

**Top LGBM2c features by split count:**

| Feature | Splits | Role |
|---------|--------|------|
| `india_vix` | 4,429 | Market fear / regime signal |
| `cpr_width_pct` | ~3,200 | Pivot compression |
| `vol_rank` | ~2,800 | Volume relative strength |
| `sg_vel` | ~2,500 | Price momentum |
| `rsi14` | ~2,300 | Overbought/oversold |

**Metrics:** Global test AUC 0.6112, recent-data AUC 0.6465, full-dataset AUC 0.6932

---

### Phase 3 — LSTM + Stacking Ensemble (Available, Bypassed)

Phase 2c alone outperforms the stack (AUC 0.6932 vs 0.6116). Meta-stacker available but bypassed.

---

### Phase 4 — PPO Position Sizing (Pure JS as of v2.5)

PPO policy runs in Node.js via `lib/lgbmInfer.js`. No Python dependency at runtime.

**State (60-dim):** 56 base features + `[lgbm2c_score, exposure, cumulative_pnl, win_streak]`

**Action:** Discrete(5) → position size ∈ {0%, 25%, 50%, 75%, 100%}

---

## Win Rate Ceiling

Structural ceiling from backtest (CPR daily rules, 5-bar hold): **~54–55%**.

| Condition | Win Rate | Signal count |
|-----------|----------|--------------|
| Baseline (all signals) | ~50% | ~1,414 stocks × all rules |
| conf ≥ 0.62 (LGBM2c) | 54.4% | Reduced — high precision subset |
| rule4 + rule11 + VIX | 54.4% | High Yield subset |

75%+ win rate is structurally impossible at daily CPR rules without look-ahead. The edge is EV: +1.98% mean return at conf≥0.62 vs +0.34% baseline.

---

## Repository Structure

```
nse-screener/
├── server.js                      # Express SSE server (scan engine)
├── ml_engine.js                   # HMM Viterbi + feature scoring
├── lib/
│   └── lgbmInfer.js               # Pure-JS LGBM2c + PPO inference
├── public/
│   └── index.html                 # Single-page screener UI
├── models/                        # Trained model artifacts
│   ├── lgbm2c_global.txt          # Phase 2c global (63 features)
│   ├── lgbm2c_regime_{0-3}.txt    # Phase 2c per-regime models
│   ├── lgbm_model.txt             # Phase 1 global (38 features)
│   ├── lgbm_regime_{0-3}.txt      # Phase 1 regime sub-models
│   ├── lgbm_rule{1-11}.txt        # Phase 1 rule sub-models
│   ├── hmm_params.json            # HMM matrices + regime map
│   ├── ppo_policy_weights.json    # PPO weights (JS-loadable)
│   ├── shap_gate_weights.json     # Rule gate weights from SHAP
│   └── signal_dataset.csv         # ⚠ gitignored — ~400 MB
├── scripts/
│   └── ml/
│       ├── build_dataset.py       # Build signal dataset (43 features)
│       ├── train_phase1.py        # HMM + LGBM pipeline
│       ├── train_phase2.py        # SHAP + conformal calibration
│       ├── train_phase3.py        # LSTM + stacking (optional)
│       ├── train_phase4.py        # PPO position sizing
│       ├── run_all.py             # Full pipeline orchestrator
│       ├── data_utils.py          # Feature schema + TA helpers
│       ├── download_bhavcopy.py   # NSE delivery % download
│       └── download_pcr.py        # NSE FO PCR download
├── kaggle/
│   ├── phase2c/
│   │   └── cpr_phase2c_kernel.py  # Phase 2c Kaggle kernel
│   └── phase4/
│       └── cpr_phase4_kernel.py   # Phase 4 Kaggle kernel
├── requirements.txt               # Python dependencies (training only)
├── package.json                   # Node dependencies
└── vercel.json                    # Vercel deployment config
```

---

## Setup

### Prerequisites

- Node.js 18+ (runtime — all inference is pure JS)
- Python 3.10+ (training only — not needed to run the screener)
- NSE Nifty 500+ OHLCV CSV

### 1. Install dependencies

```bash
npm install
```

### 2. Start the screener

```bash
npm run dev
```

Open `http://localhost:3000`.

### 3. Deploy to Vercel

```bash
npx vercel --prod --yes
```

No environment variables required for the ML pipeline — models load from `models/` at startup.

---

## Retraining (Optional)

Monthly refresh:

```bash
pip install -r requirements.txt

python scripts/ml/download_bhavcopy.py   # extend delivery data
python scripts/ml/download_pcr.py        # extend PCR data
python scripts/ml/build_dataset.py       # rebuild signal dataset
python scripts/ml/run_all.py --skip-dataset   # retrain all phases
```

A Windows Task Scheduler task (`UC_XGB_AutoRetrain`) fires monthly (next: 2026-09-16) to automate steps 3–4.

**Kaggle GPU training** (recommended for Phase 2c):

```bash
# Phase 2c: Regime-conditional LGBM (63 features, ~3 hr on T4)
# See kaggle/phase2c/cpr_phase2c_kernel.py
```

---

## Performance Metrics (v2.5, OOS holdout)

| Model | AUC | Notes |
|-------|-----|-------|
| LightGBM Phase 1 (global) | 0.571 | 38 features |
| Regime soft-blend | 0.644 | 4-state posterior-weighted |
| XGBoost Phase 2 | 0.619 | 38 features |
| LightGBM Phase 2b | 0.612 | 100-trial Optuna HPO |
| **Phase 2c (global test)** | **0.611** | **63 features — primary scorer** |
| Phase 2c (recent data) | 0.647 | Last 20% temporal split |
| Phase 2c (full dataset) | 0.693 | All data |
| Phase 3 stack | 0.612 | Bypassed — hurts AUC |
| Backtest EV at conf≥0.62 | +1.98% | India VIX + LGBM2c scoring |

---

## Technical Notes

### Pure-JS Inference (v2.5)

`lib/lgbmInfer.js` ports the LGBM2c tree ensemble to JavaScript. Leaf values and split conditions are serialised from the trained `.txt` model. PPO policy weights are loaded from `ppo_policy_weights.json`. This eliminates the Flask microservice and cold-start latency on Vercel.

### Circuit-Day CPR Fix

When NSE imposes a circuit limit (H = L = C), the resulting CPR is degenerate (upper = lower). `getPrev()` skips these bars in the prevDay lookup; `pPos()` handles the degenerate case to avoid false "Inside CPR" signals.

### India VIX Integration

Fetched from Yahoo Finance (`^INDIAVIX`) at scan start. Zero-VIX guard prevents division errors; scores gracefully degrade when VIX is unavailable (LGBM2c returns lower confidence rather than crashing).

### Windows OOM Fix (Training)

After HMM training, Windows heap fragmentation prevents LightGBM from allocating contiguous blocks. Phase 1B runs in a fresh subprocess (`--lgbm-only` flag). `memory_map=True` in pandas avoids the large contiguous buffer required by the C parser.

### PCR Coverage

Per-symbol stock options PCR covers ~15–30% of signal dates. Fallback: market-wide Nifty OPTIDX PCR stored as `__MKT_NIFTY__` — near-100% date coverage.

---

## License

**Proprietary — All Rights Reserved.**

This software and all associated algorithms, ML models, and trading logic are the exclusive intellectual property of the copyright holder. No license is granted to use, copy, modify, distribute, or create derivative works. See [LICENSE](LICENSE) for full terms.

For licensing inquiries: [github.com/Kasi72](https://github.com/Kasi72)

---

*Built for NSE India intraday trading research. Not financial advice.*
