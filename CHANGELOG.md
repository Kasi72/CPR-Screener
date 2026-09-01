# Changelog

All notable changes to NSE CPR Screener are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Versioning policy**  
> `MAJOR` — breaking change to API contracts, model schema, or FEATURE_COLS ordering  
> `MINOR` — new features, new model phases, backward-compatible additions  
> `PATCH` — bug fixes, data pipeline fixes, documentation, performance tuning

---

## [2.0.0] — 2026-09-01

### Breaking Changes

- **FEATURE_COLS expanded from 25 → 38 features.** Models trained on v1.x are incompatible with v2.x inference. A full retrain (`python scripts/ml/run_all.py`) is required after upgrading.
- **MONOTONE_CONSTRAINTS** updated to 38 values — any hard-coded constraint lists in external code must be regenerated from `data_utils.MONOTONE_CONSTRAINTS`.
- `calc_cpr()` now returns two additional keys (`r1`, `s1`). Code that destructures the return dict by position will break (use keyword access).
- `build_signals_for_symbol()` adds a new `market_pcr` parameter — callers must pass or accept `**kwargs`.

### Added

#### Feature Engineering (Tier 2) — 13 new features

- **`cpr_compress`** — Ratio of today's CPR width to 5-day average CPR width. Values below 1.0 indicate CPR squeeze, historically a precursor to range expansion and breakout signals.
- **`cpr_pos`** — Normalised close position within CPR band `[0 = at lower band, 1 = at upper band]`. Encodes whether price is accepting or rejecting the CPR range.
- **`dist_r1`** — Close distance from classic floor-pivot R1 level. Negative = below R1, positive runway available.
- **`dist_s1`** — Close distance from S1 pivot. Positive = above S1, support confirmed beneath signal.
- **`mom3`**, **`mom10`**, **`mom20`** — Multi-timeframe momentum returns (3 / 10 / 20 bars). Jointly encode momentum alignment across short, medium, and intermediate lookbacks.
- **`rsi_div`** — RSI divergence feature: `+1` when price fell but RSI rose over 5 bars (bullish divergence), `−1` when price rose but RSI fell (bearish divergence), `0` otherwise. Threshold: |price_ret| > 0.5%, |rsi_chg| > 1.0.
- **`vol_accel_delta`** — Change in volume surge ratio vs prior day. Captures the *acceleration* of institutional participation, not just its level.
- **`days_since_52hi`** — Calendar days since the stock last traded at its 52-week high. Low values indicate recent momentum; high values indicate prolonged underperformance.
- **`expiry_dist`** — Calendar days to the next monthly F&O expiry (last Thursday of the month). Captures options-cycle effects on volatility and institutional hedging flows.
- **`regime_stability`** — `max_posterior_today − max_posterior_yesterday` derived from HMM posteriors. Positive = regime conviction increasing; negative = regime fracturing.
- **`transition_risk`** — `1 − max_posterior`. Near zero = high regime certainty; near 0.75 = regime ambiguity, signal quality lower.

#### Model Architecture

- **Per-rule LightGBM sub-models** (`lgbm_rule1.txt` … `lgbm_rule11.txt`). Each of the 11 CPR/Camarilla rules now has a dedicated model capturing its specific win pattern, position in the pivot structure, and directional bias. At inference, predictions are blended: 70% global LGBM + 30% rule-specific.
- **Rule AUC tracking** added to `phase1_metrics.json` under `rule_aucs` key.
- `predict_server.py` loads rule sub-models at startup and applies per-rule blending in `/predict_ensemble`.

#### Data Pipeline

- **Nifty OPTIDX market PCR** — `download_pcr.py` now also parses index options (OPTIDX: NIFTY, BANKNIFTY, FINNIFTY) from the same FO bhavcopy file. Stored as `__MKT_NIFTY__` symbol. Achieves near-100% date coverage vs ~20% for per-symbol stock options.
- **`load_market_pcr()`** — new function returning a date-indexed `pd.Series` of Nifty OPTIDX PCR for use as a market-wide fallback in `build_dataset.py`.
- `build_dataset.py` now falls back to market PCR when per-symbol stock PCR is missing for a given date, significantly improving PCR feature coverage.
- `build_signals_for_symbol()` accepts `market_pcr` parameter.

#### HMM Regime Quality

- `train_phase1.py` joins `regime_stability` and `transition_risk` directly into the training DataFrame from `hmm_posteriors.json` using vectorised `pd.Series.map`, making HMM quality features available to all downstream model phases without re-running the HMM.

#### Infrastructure

- **`expiry_dist_days(dt)`** helper added to `data_utils.py` — returns calendar days to the next last-Thursday-of-month F&O expiry.
- `calc_cpr()` extended with `r1` and `s1` floor-pivot keys.
- `build_features()` in `data_utils.py` updated with neutral defaults for all 13 new features.
- `predict_server.py` `_FEATURE_DEFAULTS` extended; `RULE_PATHS` dict added for model path resolution.
- `.gitignore` extended to exclude `signal_dataset.csv` (406 MB), `delivery_data.pkl` (88 MB), `xgb_v2_features.csv`, and other large generated artifacts.

### Changed

- `load_pcr_data()` now filters out `__MKT_*` rows before building the per-symbol pivot table, preserving backward compatibility.
- `phase1_metrics.json` schema updated: top-level key `rule_aucs` added alongside existing `regime_aucs`.
- `train_phase2.py` comment updated: references 38-feature schema.
- All training phases now call `for col in FEATURE_COLS: df[col] = df.get(col, 0.0)` before feature matrix construction — graceful fallback for datasets built with older versions of `build_dataset.py`.

### Fixed

- `rsi_div` computation correctly handles edge cases where `i < 34` (insufficient history for 5-bar-lagged RSI comparison).
- `expiry_dist_days` iterates up to 3 months forward to handle date inputs on or after the monthly expiry day.

---

## [1.4.0] — 2026-08-12

### Added

- **Tier 1 interaction features** (`conf_vol`, `rsi_dir`, `hi52_dir`) bringing total features to 25.
- `india_vix` as direct input to LightGBM (feature 22).
- Nifty regime gate: signals in `High-Vol-Panic` regime are suppressed at the screener UI layer.
- `brain_goldmine.js` forward-labelled outcome (`hit_t1`) validation run.

### Changed

- UC score weights dynamically recomputed: `VolPre5` d=1.21, `RangeATR` d=1.05, `RSI2` d=0.02.
- Live `ucScoreWeights.ts` updated from static defaults to data-derived weights.

### Fixed

- `rule_id` float32 cast error in `load_signal_dataset` — `rule_id` column excluded from dtype dict since it stores strings `'rule1'…'rule11'`.

---

## [1.3.0] — 2026-08-08

### Added

- **Regime-specific LightGBM sub-models** (4 states: Bull-Trend, Bear-Trend, Chop, High-Vol-Panic) with per-regime Optuna HPO (20 trials each).
- **Posterior soft-blending**: `Σ P(state|obs) × P(win|x, state)` replaces hard regime routing. Soft-blend AUC reported separately.
- `hmm_posteriors.json` saved during Phase 1A — date-keyed posterior distributions used for soft blending at inference.
- `soft_blend_config.json` model artefact.
- Phase 1B subprocess isolation (`--lgbm-only` flag + `subprocess.run`) to prevent Windows heap OOM after HMM training.

### Fixed

- **Windows OOM after HMM**: `gc.collect()` alone insufficient — OS does not reclaim fragmented C heap pages. Solution: LightGBM runs in a fresh subprocess, letting the OS fully reclaim all HMM pages before the CSV load.
- `memory_map=True` in `load_signal_dataset` — replaces Python-engine chunked reader that triggered `OSError: [Errno 22] Invalid argument` on large Windows files.

---

## [1.2.0] — 2026-08-05

### Added

- **Phase 4: PPO position sizing** (Stable-Baselines3, custom Gymnasium environment).
- `ppo_position_size()` scoring helper in `scoring.py`.
- `/position_size` endpoint in `predict_server.py`.

### Changed

- `run_all.py` now orchestrates all 4 phases as separate subprocesses.
- Phase skip flags: `--skip-phase1` through `--skip-phase4`.

---

## [1.1.0] — 2026-07-28

### Added

- **Phase 3: LSTM + stacking ensemble.** 2-layer LSTM on 20-bar sequences, logistic meta-learner.
- `lstm_model.py` — `LSTMSignalModel` (PyTorch).
- `/predict_ensemble` endpoint with XGB + LGBM + LSTM + regime stacking.
- Conformal prediction intervals (`lo`, `hi`) returned by ensemble endpoint.
- **Phase B: market and sector relative strength** (4 features: `market_rs_{5d,20d}`, `sector_rs_{5d,20d}`).
- **Phase C: NSE delivery %** (`deliv_pct`) from bhavcopy.
- **Phase D: options put-call ratio** (`pcr`) from FO bhavcopy.

### Changed

- `FEATURE_COLS` expanded to 21 features.

---

## [1.0.0] — 2026-07-15

### Added

- Initial release. NSE CPR screener with 11-rule signal engine (CPR × Camarilla) and confluence gate (≥ 2 rules required).
- **Phase 1: HMM + LightGBM** — 4-state Gaussian HMM on Nifty 50, global LightGBM with Optuna HPO.
- **Phase 2: SHAP gate weights + conformal calibration** — SHAP importance → rule gate weights (0.5–2.0), nonconformity calibration.
- 12 baseline features: CPR width, VWAP dist, ATR rank, vol rank, rules fired, SG velocity, EMA200 dist, RSI14, 5-day momentum, day-of-week, rule ID, direction.
- **Phase A: 52-week context** (`dist_hi52`, `dist_lo52`, `vol_accel`).
- `predict_server.py` Flask microservice — `/predict`, `/regime`, `/gate_weights`, `/health`.
- `server.js` Express frontend with Vercel deployment.
- `public/index.html` single-page screener UI.
- `ml_engine.js` in-browser HMM Viterbi decoder.

---

[2.0.0]: https://github.com/Kasi72/CPR-Screener/compare/v1.4.0...v2.0.0
[1.4.0]: https://github.com/Kasi72/CPR-Screener/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/Kasi72/CPR-Screener/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/Kasi72/CPR-Screener/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/Kasi72/CPR-Screener/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Kasi72/CPR-Screener/releases/tag/v1.0.0
