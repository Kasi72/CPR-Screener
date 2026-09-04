# Changelog

All notable changes to NSE CPR Screener are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Versioning policy**  
> `MAJOR` — breaking change to API contracts, model schema, or FEATURE_COLS ordering  
> `MINOR` — new features, new model phases, backward-compatible additions  
> `PATCH` — bug fixes, data pipeline fixes, reward function tuning, performance fixes

---

## [Unreleased]

---

## [2.4.1] — 2026-09-04

### Fixed

- **Phase 4 PPO reward reverted to P&L.** The Sharpe-delta reward introduced in v2.4.0 produced high gradient variance and consistently negative delta (−0.038 on the OOS holdout) — the rolling Sharpe changed too slowly to provide useful per-step feedback. Reward is now `trade_return × size_frac − 0.001 × |action_delta|` (direct P&L minus transaction cost), identical to the original reward used before the Sharpe-delta experiment.

### Changed

- `cpr_phase4_kernel.py`: removed `_rolling_sharpe` helper and `SHARPE_WINDOW` constant; replaced `running_sharpe` state slot with `cum_pnl`; reward now uses direct P&L.
- `SignalSizingEnv` docstring updated to reflect active reward design.

---

## [2.4.0] — 2026-09-04

### Added

- **Phase 4 PPO v2 — `lgbm2c_score` in state.** The PPO state vector now includes `lgbm2c_score` (the Phase 2c ML confidence score) as the primary signal-quality input, replacing the earlier `regime_score` slot. This gives the agent direct access to the full 63-feature ML estimate without having to re-derive it from the base state features.
- **200k training timesteps.** PPO training budget doubled from 100k to 200k timesteps with early-stopping (8 consecutive evaluations without improvement, evaluated every 5k steps). Net effect: ~2.5× wall-clock on Kaggle T4 but better convergence.
- **Phase 4 Kaggle automation.** `kaggle_phase4_runner.py` — orchestrates dataset upload, kernel push, status polling, and output download. `post_phase4_deploy.py` — background watcher that triggers `git commit + vercel --prod` the moment Phase 4 output files appear in `models/`.

### Changed

- `STATE_DIM` updated from 5 to 60 (56 signal base features + `lgbm2c_score` + `exposure` + `cum_pnl/running_sharpe` + `win_streak`).
- `FEATURE_COLS` in `cpr_phase4_kernel.py` aligned with Phase 2c training schema (56 features including Sprint 1–2B CPR depth features and Sprint 3 features).
- `SignalSizingEnv.reset()` initialises `cum_pnl` / `running_sharpe` slot.

### Fixed

- `n_eval_episodes` reduced from 10 → 1 and `eval_freq` from 10k → 5k, eliminating 146× overhead in the evaluation callback loop.
- `post_phase4_deploy.py` startup-banner false positive: watcher now baselines file mtime at launch so it ignores pre-existing output files from prior runs.
- `post_phase4_deploy.py` Kaggle timeout handling: on local poll timeout the script falls back to `kaggle kernels output` download rather than aborting.

---

## [2.3.0] — 2026-09-03

### Added

- **Sprint 3 features** — 5 new features added to `build_dataset.py` and `data_utils.FEATURE_COLS`:
  - `gap_pct` — open-gap percentage vs prior close; captures overnight sentiment impulse.
  - `cpr_test_count_5d` — number of times price tested the CPR band in the prior 5 sessions; high count = well-established support/resistance.
  - `prev_bar_close_pos` — prior bar close position relative to CPR midpoint (normalised); encodes intraday continuation context.
  - `atr_expansion` — ATR today / 10-day ATR average; flags volatility regime shifts.
  - `vol_trend_slope` — 20-day linear slope of normalised volume; captures rising vs declining participation.
- **`lgbm2c_score` injection** — `scripts/ml/score_p2c.py` scores all 1.39M signals with the Phase 2c model and writes `lgbm2c_score` into `signal_dataset.csv`, making it available as a feature for Phase 4 PPO training.
- **Phase 3 ablation confirmed.** Meta-stacking formally benchmarked: Phase 2c alone AUC 0.6465 (recent) / 0.6932 (full); Phase 3 stack AUC 0.6116. Phase 2c bypasses the meta-stacker in `predict_server.py`.
- `predict_server.py` routing: `stack_score` now returns `lgbm2c_score` directly; `meta_lgbm` loaded lazily for research parity only.

### Changed

- `data_utils.FEATURE_COLS` extended from 38 to 43 features.
- `predict_server.py` `extract_features_2c()`: updated to 63-feature extraction including Sprint 3 features and all 7 interaction features.
- `predict_server.py` `score_lgbm2c()`: fixed interaction formulas to match `score_p2c.py` exactly; fixed direction-adjustment feature set membership.
- `predict_server.py` `score_lgbm2c()`: inference blend corrected to 50% global + 50% regime (was regime-only).
- `score_p2c.py`: `prev_bar_close_pos` excluded from `DIRECTIONAL_FEATURES` (semantically ambiguous for SELL signals).
- Phase 3 Kaggle runner: `lstm_model.pt` marked optional; 90-second dataset commit wait added.
- Phase 2c Kaggle runner: `DIRECTIONAL_FEATURES` fix propagated.

---

## [2.2.0] — 2026-09-02

### Added

- **Sprint 2A CPR structure features** (5 new features):
  - `open_inside_cpr` — binary: open price inside CPR band. Strong bias toward range-bound day.
  - `cpr_virgin` — binary: CPR has not yet been touched by price this session.
  - `consecutive_narrow_cprs` — count of consecutive days with CPR width below the 10th percentile; multi-day squeeze accumulates breakout energy.
  - `cpr_midpoint_trend` — 5-day slope of CPR midpoint; rising = bullish drift.
  - `cpr_expansion_factor` — today's CPR width / prior-day width; values > 1 flag sudden range expansion.

- **Sprint 2B CPR context features** (5 new features):
  - `cpr_above_prev_cpr` — binary: today's CPR midpoint above yesterday's. Bullish pivot drift.
  - `prev_close_inside_cpr` — binary: prior day close inside today's CPR band; strong magnetic effect.
  - `atr_to_cpr_ratio` — ATR / CPR width; values < 1 flag CPR too wide relative to typical range.
  - `cpr_width_percentile_252d` — CPR width percentile rank over prior 252 trading days. Historical squeeze context.
  - `prev_day_ochoa_type` — prior-day CPR type (neutral/wide/trending) as ordinal integer.

- **Phase 2c Kaggle kernel** (`kaggle/phase2c/cpr_phase2c_kernel.py`):
  - Trains 1 global + 4 per-regime LightGBM models on 63 features (56 base + 7 interactions).
  - 7 interaction features: `cpr_vol_interaction`, `regime_momentum`, `cpr_rsi_squeeze`, `overlap_vol_signal`, `rs_direction_alignment`, `virgin_momentum`, `narrow_breakout_vol`.
  - Regime-conditional training: each regime sub-model uses Optuna HPO tuned for its data distribution.
  - Outputs: `lgbm2c_global.txt`, `lgbm2c_regime_{0-3}.txt`, `phase2c_metrics.json`, `shap_weights2c.json`.

- **Live price refresh UI.** `server.js` adds `/api/live-prices` endpoint that fetches current prices from NSE. `public/index.html` adds a "Refresh Prices" button that updates the screener table in-place without a full reload.

### Changed

- Phase 2c supersedes Phase 2b as the primary LGBM scorer. `predict_server.py` updated to load `lgbm2c_global.txt` and regime models.
- `build_dataset.py` updated with Sprint 2A+2B feature computation logic.

---

## [2.1.0] — 2026-09-01

### Added

- **Sprint 1 CPR depth features** (5 new features added to Phase 2c training schema):
  - `cpr_overlap_pct` — overlap between today's and yesterday's CPR as % of today's width. High overlap = strong pivot continuity.
  - `open_to_cpr_dist` — open price distance from CPR midpoint, signed. Inside CPR = 0 boundary; above/below encodes directional bias.
  - `prev_cpr_respected` — binary: prior session price respected the CPR as support/resistance. Qualifies the CPR's predictive power.
  - `cpr_zone_vol_ratio` — ratio of volume traded within the CPR zone to total day volume. High ratio = CPR is the key acceptance/rejection zone.
  - `hmm_regime` — HMM regime integer (0–3) as a categorical feature, enabling regime-conditional splits within a single decision tree.

- **Phase 2b — LightGBM HPO (Kaggle).** `kaggle_phase2b_runner.py` + `kaggle/phase2b/cpr_phase2b_kernel.py`. 100-trial Optuna optimisation on 1.39M signals, 38 features. CV AUC 0.5958, Val AUC 0.6124.

- **Phase 4 initial PPO Kaggle pipeline.** `kaggle_phase4_runner.py` + `kaggle/phase4/cpr_phase4_kernel.py`. Custom Gymnasium environment `SignalSizingEnv` with Discrete(5) action space (position size tiers) and P&L reward. Trained with Stable-Baselines3 PPO, 100k timesteps.

- **Phase 2b scorer wiring.** `predict_server.py` updated to load `lgbm2b.txt` scores. `score_p2b.py` script injects per-signal `lgbm2b_score` into `signal_dataset.csv` for Phase 3/4 downstream use.

- **Post-training deploy pipeline.** `scripts/ml/post_phase4_deploy.py` background watcher: polls `models/` for Phase 4 output files, then runs `git commit + vercel --prod` automatically.

- **Scientifically calibrated recommendation weights.** `calibrate_rec_weights.py` fits logistic regression from UC-score components to `hit_t1` outcomes. Weights written to `models/rec_weights.json`.

### Changed

- `kaggle_phase3_runner.py`: `lstm_model.pt` optional; 90-second dataset-version commit wait added before kernel push.
- `train_phase4.py` (local): inherits Phase 2b best params for LightGBM baseline.

### Fixed

- Phase 4 eval overhead: `n_eval_episodes` reduced 10 → 1, `eval_freq` adjusted from 10k → 2048 steps (146× speedup).
- Watcher false positive on startup banner containing the substring `"timeout"` — now uses exact state string matching.
- Kaggle Phase 3 CUDA probe: added CPU fallback when T4 is not available for the LSTM probe step.
- Kaggle Phase 3 kernel slug corrected; dataset indexing wait (90s) added after upload.
- Phase 3 kernel: stacking feature-count mismatch (25 vs 38) fixed by aligning `META_FEATURES` with live `FEATURE_COLS`.

---

## [2.0.0] — 2026-09-01

### Breaking Changes

- **FEATURE_COLS expanded from 25 → 38.** Models trained on v1.x are incompatible with v2.x inference. Full retrain required.
- **MONOTONE_CONSTRAINTS** updated to 38 values — any hard-coded constraint lists in external code must be regenerated.
- `calc_cpr()` now returns two additional keys (`r1`, `s1`). Code that destructures by position will break.
- `build_signals_for_symbol()` adds `market_pcr` parameter — callers must pass or accept `**kwargs`.

### Added

#### Feature Engineering — 13 new features (Phase D tier)

- `cpr_compress` — CPR width / 5-day average CPR width. Values < 1 indicate squeeze preceding breakout.
- `cpr_pos` — Close normalised within CPR band [0 = lower, 1 = upper].
- `dist_r1`, `dist_s1` — Close distance from R1 and S1 floor-pivot levels.
- `mom3`, `mom10`, `mom20` — Multi-timeframe momentum (3 / 10 / 20 bars).
- `rsi_div` — RSI divergence (+1 bullish / −1 bearish / 0 none).
- `vol_accel_delta` — Change in volume surge ratio vs prior day.
- `days_since_52hi` — Calendar days since last 52-week high.
- `expiry_dist` — Days to next monthly F&O expiry (last Thursday of month).
- `regime_stability` — HMM max-posterior delta (day-over-day conviction change).
- `transition_risk` — `1 − max_posterior` (regime ambiguity proxy).

#### Model Architecture

- Per-rule LightGBM sub-models (11 models): `lgbm_rule1.txt` … `lgbm_rule11.txt`. At inference: 70% global + 30% rule-specific blend.
- Rule AUC tracking in `phase1_metrics.json` under `rule_aucs` key.
- `predict_server.py` loads rule sub-models and applies per-rule blending in `/predict_ensemble`.

#### Data Pipeline

- Nifty OPTIDX market PCR: `download_pcr.py` parses index options (OPTIDX: NIFTY, BANKNIFTY, FINNIFTY). Stored as `__MKT_NIFTY__` symbol. Near-100% date coverage vs ~20% for per-symbol OPTSTK.
- `load_market_pcr()` — date-indexed Series of Nifty OPTIDX PCR for fallback.
- `build_dataset.py` falls back to market PCR when per-symbol stock PCR is missing.
- `build_signals_for_symbol()` accepts `market_pcr` parameter.

#### Infrastructure

- `expiry_dist_days(dt)` helper — returns calendar days to next last-Thursday-of-month F&O expiry.
- `calc_cpr()` extended with `r1`, `s1` floor-pivot keys.
- `build_features()` updated with neutral defaults for all 13 new features.
- `predict_server.py` `_FEATURE_DEFAULTS` extended; `RULE_PATHS` dict added.
- `.gitignore` excludes `signal_dataset.csv` (~406 MB), `delivery_data.pkl`, `xgb_v2_features.csv`.

### Changed

- `load_pcr_data()` filters out `__MKT_*` rows before per-symbol pivot table construction.
- `phase1_metrics.json` schema: `rule_aucs` key added.
- All training phases apply `df[col] = df.get(col, 0.0)` fallback for datasets built with older `build_dataset.py`.

### Fixed

- `rsi_div` computation handles edge cases where `i < 34` (insufficient history for 5-bar RSI comparison).
- `expiry_dist_days` iterates up to 3 months forward to handle inputs on or after the monthly expiry day.

---

## [1.4.0] — 2026-08-12

### Added

- Tier 1 interaction features (`conf_vol`, `rsi_dir`, `hi52_dir`) bringing total to 25.
- `india_vix` as direct LightGBM input (feature 22).
- Nifty regime gate: `High-Vol-Panic` signals suppressed at the UI layer.
- `brain_goldmine.js` forward-labelled outcome (`hit_t1`) validation run.

### Changed

- UC score weights dynamically recomputed: `VolPre5` d=1.21, `RangeATR` d=1.05, `RSI2` d=0.02.
- `ucScoreWeights.ts` updated from static defaults to data-derived weights.

### Fixed

- `rule_id` float32 cast error in `load_signal_dataset` — `rule_id` column excluded from dtype dict.

---

## [1.3.0] — 2026-08-08

### Added

- Regime-specific LightGBM sub-models (4 states) with per-regime Optuna HPO (20 trials each).
- Posterior soft-blending: `Σ P(state|obs) × P(win|x, state)` replaces hard regime routing.
- `hmm_posteriors.json` saved during Phase 1A.
- `soft_blend_config.json` model artifact.
- Phase 1B subprocess isolation (`--lgbm-only` flag) to prevent Windows heap OOM.

### Fixed

- **Windows OOM after HMM**: `gc.collect()` insufficient — OS does not reclaim fragmented C heap pages. LightGBM now runs in a fresh subprocess, reclaiming all HMM pages first.
- `memory_map=True` in `load_signal_dataset` — replaces Python-engine chunked reader that triggered `OSError: [Errno 22] Invalid argument` on large Windows files.

---

## [1.2.0] — 2026-08-05

### Added

- Phase 4: PPO position sizing (Stable-Baselines3, custom Gymnasium environment).
- `ppo_position_size()` scoring helper in `scoring.py`.
- `/position_size` endpoint in `predict_server.py`.

### Changed

- `run_all.py` orchestrates all 4 phases as separate subprocesses.
- Phase skip flags: `--skip-phase1` through `--skip-phase4`.

---

## [1.1.0] — 2026-07-28

### Added

- Phase 3: LSTM + stacking ensemble. 2-layer LSTM on 20-bar sequences, logistic meta-learner.
- `lstm_model.py` — `LSTMSignalModel` (PyTorch).
- `/predict_ensemble` endpoint with XGB + LGBM + LSTM + regime stacking.
- Conformal prediction intervals (`lo`, `hi`) returned by ensemble endpoint.
- Phase B: market and sector relative strength (4 features: `market_rs_{5d,20d}`, `sector_rs_{5d,20d}`).
- Phase C: NSE delivery % (`deliv_pct`) from bhavcopy.
- Phase D: options put-call ratio (`pcr`) from FO bhavcopy.

### Changed

- `FEATURE_COLS` expanded to 21 features.

---

## [1.0.0] — 2026-07-15

### Added

- Initial release. NSE CPR screener with 11-rule signal engine (CPR × Camarilla) and confluence gate (≥ 2 rules required).
- Phase 1: HMM + LightGBM — 4-state Gaussian HMM on Nifty 50, global LightGBM with Optuna HPO.
- Phase 2: SHAP gate weights + conformal calibration.
- 12 baseline features: CPR width, VWAP dist, ATR rank, vol rank, rules fired, SG velocity, EMA200 dist, RSI14, 5-day momentum, day-of-week, rule ID, direction.
- Phase A: 52-week context (`dist_hi52`, `dist_lo52`, `vol_accel`).
- `predict_server.py` Flask microservice — `/predict`, `/regime`, `/gate_weights`, `/health`.
- `server.js` Express frontend with Vercel deployment.
- `public/index.html` single-page screener UI.
- `ml_engine.js` in-browser HMM Viterbi decoder.

---

[Unreleased]: https://github.com/Kasi72/CPR-Screener/compare/v2.4.1...HEAD
[2.4.1]: https://github.com/Kasi72/CPR-Screener/compare/v2.4.0...v2.4.1
[2.4.0]: https://github.com/Kasi72/CPR-Screener/compare/v2.3.0...v2.4.0
[2.3.0]: https://github.com/Kasi72/CPR-Screener/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/Kasi72/CPR-Screener/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/Kasi72/CPR-Screener/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/Kasi72/CPR-Screener/compare/v1.4.0...v2.0.0
[1.4.0]: https://github.com/Kasi72/CPR-Screener/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/Kasi72/CPR-Screener/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/Kasi72/CPR-Screener/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/Kasi72/CPR-Screener/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Kasi72/CPR-Screener/releases/tag/v1.0.0
