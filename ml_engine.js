'use strict';
/**
 * ml_engine.js — JavaScript ML inference layer (pure-JS, no Python subprocess)
 *
 * Responsibilities:
 *   1. HMM Viterbi decoder (from hmm_params.json)
 *   2. Conformal prediction interval (from conformal_scores.json)
 *   3. LGBM2c ensemble score — pure JS, from lgbm2c_global_js.json + lgbm2c_regime_N_js.json
 *   4. PPO position size — pure JS, from ppo_policy_weights.json
 *   5. SHAP gate weight loader (from shap_gate_weights.json)
 */

const fs   = require('fs');
const path = require('path');
const { loadModel, predictOne } = require('./lib/lgbmInfer');
const { ppoPredict }             = require('./lib/ppoInfer');

const MODELS_DIR        = path.join(__dirname, 'models');
const LGBM_GLOBAL_PATH  = path.join(MODELS_DIR, 'lgbm2c_global_js.json');
const LGBM_REGIME_PATHS = [0, 1, 2, 3].map(i => path.join(MODELS_DIR, `lgbm2c_regime_${i}_js.json`));
const PPO_WEIGHTS_PATH  = path.join(MODELS_DIR, 'ppo_policy_weights.json');

// ─── Feature schema (must match _P2C_ALL_FEATURES in predict_server.py) ──────
const _P2C_BASE_FEATURES = [
  'cpr_width_pct', 'vwap_dist',    'atr_pct_rank',  'vol_rank',
  'n_rules_fired', 'sg_vel',       'ema200_dist',    'rsi14',
  'mom5',          'dow',          'rule_id',         'direction',
  'dist_hi52',     'dist_lo52',    'vol_accel',
  'market_rs_5d',  'market_rs_20d','sector_rs_5d',   'sector_rs_20d',
  'deliv_pct',     'pcr',          'india_vix',
  'conf_vol',      'rsi_dir',      'hi52_dir',
  'cpr_compress',  'cpr_pos',      'dist_r1',         'dist_s1',
  'mom3',          'mom10',        'mom20',
  'rsi_div',       'vol_accel_delta',
  'days_since_52hi','expiry_dist',
  'cpr_overlap_pct','open_to_cpr_dist','prev_cpr_respected','cpr_zone_vol_ratio',
  'hmm_regime',
  'open_inside_cpr','cpr_virgin','consecutive_narrow_cprs',
  'cpr_midpoint_trend','cpr_expansion_factor',
  'cpr_above_prev_cpr','prev_close_inside_cpr','atr_to_cpr_ratio',
  'cpr_width_percentile_252d','prev_day_ochoa_type',
  'gap_pct','cpr_test_count_5d','prev_bar_close_pos',
  'atr_expansion','vol_trend_slope',
];  // 56

// Features whose sign gets flipped for SELL direction (matches predict_server.py)
const _P2C_DIRECTIONAL = new Set([
  'dist_hi52','dist_lo52','vwap_dist','ema200_dist',
  'mom3','mom5','mom10','mom20',
  'market_rs_5d','market_rs_20d','sector_rs_5d','sector_rs_20d',
  'cpr_pos','dist_r1','dist_s1','sg_vel',
  'open_to_cpr_dist','gap_pct',
]);

// Defaults for features that may be absent (matches _P2C_DEFAULTS in predict_server.py)
const _P2C_DEFAULTS = {
  cpr_overlap_pct: 0.5, open_to_cpr_dist: 0.0, prev_cpr_respected: 0.0,
  cpr_zone_vol_ratio: 1.0, hmm_regime: -1, open_inside_cpr: 0.0,
  cpr_virgin: 0.0, consecutive_narrow_cprs: 0.0, cpr_midpoint_trend: 0.0,
  cpr_expansion_factor: 1.0, cpr_above_prev_cpr: 0.0, prev_close_inside_cpr: 0.0,
  atr_to_cpr_ratio: 1.0, cpr_width_percentile_252d: 0.5, prev_day_ochoa_type: 0.0,
  gap_pct: 0.0, cpr_test_count_5d: 0.0, prev_bar_close_pos: 0.5,
  atr_expansion: 1.0, vol_trend_slope: 0.0, deliv_pct: 0.0, pcr: 1.0,
};

/**
 * Build 63-element feature vector from a features object.
 * Applies directional flip on _P2C_DIRECTIONAL features, then appends 7
 * interaction features — exactly matching extract_features_2c() in predict_server.py.
 */
function buildFeatureVector(f) {
  const dir = f.direction ?? 1;

  // Resolve a single feature: apply default then directional flip
  function get(name) {
    const raw = (f[name] !== undefined && f[name] !== null) ? f[name]
              : (_P2C_DEFAULTS[name] !== undefined ? _P2C_DEFAULTS[name] : 0);
    return _P2C_DIRECTIONAL.has(name) ? raw * dir : raw;
  }

  const vec = new Float32Array(63);

  // Base 56 features
  for (let i = 0; i < 56; i++) vec[i] = get(_P2C_BASE_FEATURES[i]);

  // Interaction features (indices 56-62)
  // NOTE: Python computes interactions AFTER flipping base features in-place,
  // so get() already returns flipped values — the formulas below match Python.
  vec[56] = get('cpr_compress')   * get('vol_rank');                           // cpr_vol_interaction
  vec[57] = get('hmm_regime')     * get('mom5');                               // regime_momentum
  vec[58] = (1.0 - get('cpr_width_pct')) * get('rsi14');                      // cpr_rsi_squeeze
  vec[59] = get('cpr_overlap_pct') * get('cpr_zone_vol_ratio');                // overlap_vol_signal
  vec[60] = (get('market_rs_5d') + get('sector_rs_5d')) * dir;                // rs_direction_alignment
  vec[61] = get('cpr_virgin')     * get('mom5');                               // virgin_momentum
  vec[62] = get('consecutive_narrow_cprs') * get('vol_rank');                  // narrow_breakout_vol

  return vec;
}

// ─── State cache ──────────────────────────────────────────────────────────────
let _hmmParams          = null;
let _conformalScores    = null;
let _gateWeights        = null;
let _currentRegime      = null;
let _currentRegimeState = -1;   // HMM state integer (0=Panic,1=Bear,2=Chop,3=Bull)
let _lgbmGlobal         = null;

// ─── JSON loader helper ───────────────────────────────────────────────────────
function loadJson(filename) {
  const p = path.join(MODELS_DIR, filename);
  if (!fs.existsSync(p)) return null;
  try { return JSON.parse(fs.readFileSync(p, 'utf8')); }
  catch { return null; }
}

// ─── Initialization ───────────────────────────────────────────────────────────
function init() {
  _hmmParams       = loadJson('hmm_params.json');
  _conformalScores = loadJson('conformal_scores.json');
  _gateWeights     = loadJson('shap_gate_weights.json');

  if (_hmmParams) {
    _currentRegime = _hmmParams.current_regime || 'Unknown';
    console.log(`[ml_engine] HMM loaded. Current regime: ${_currentRegime}`);
  } else {
    console.log('[ml_engine] hmm_params.json not found — regime detection disabled');
  }
  if (_gateWeights)     console.log('[ml_engine] SHAP gate weights loaded.');
  if (_conformalScores) console.log('[ml_engine] Conformal scores loaded.');

  // Eagerly load global LGBM2c (1.76 MB, cached in lgbmInfer)
  if (fs.existsSync(LGBM_GLOBAL_PATH)) {
    try {
      _lgbmGlobal = loadModel(LGBM_GLOBAL_PATH);
      console.log(`[ml_engine] LGBM2c global loaded (${_lgbmGlobal.num_trees} trees).`);
    } catch (e) {
      console.error('[ml_engine] Failed to load lgbm2c_global_js.json:', e.message);
    }
  } else {
    console.log('[ml_engine] WARN: lgbm2c_global_js.json not found — ML scoring disabled');
  }
}

// ─── HMM Viterbi ─────────────────────────────────────────────────────────────
function gaussianLogLikelihood(means, covars, obs) {
  let ll = 0;
  for (let i = 0; i < means.length; i++) {
    const d    = obs[i] - means[i];
    const var_ = covars[i];
    ll += -0.5 * (Math.log(2 * Math.PI * var_) + d * d / var_);
  }
  return ll;
}

function viterbiDecode(obsMatrix) {
  if (!_hmmParams) return null;
  const { n_components, startprob, transmat, means, covars } = _hmmParams;
  const T = obsMatrix.length;
  const N = n_components;

  const delta = Array.from({length: T}, () => new Float64Array(N));
  const psi   = Array.from({length: T}, () => new Int32Array(N));

  for (let j = 0; j < N; j++) {
    delta[0][j] = Math.log(startprob[j] + 1e-300) +
                  gaussianLogLikelihood(means[j], covars[j], obsMatrix[0]);
  }
  for (let t = 1; t < T; t++) {
    for (let j = 0; j < N; j++) {
      let best = -Infinity, bestState = 0;
      for (let i = 0; i < N; i++) {
        const val = delta[t-1][i] + Math.log(transmat[i][j] + 1e-300);
        if (val > best) { best = val; bestState = i; }
      }
      delta[t][j] = best + gaussianLogLikelihood(means[j], covars[j], obsMatrix[t]);
      psi[t][j]   = bestState;
    }
  }

  const states = new Int32Array(T);
  let best = -Infinity;
  for (let j = 0; j < N; j++) {
    if (delta[T-1][j] > best) { best = delta[T-1][j]; states[T-1] = j; }
  }
  for (let t = T-2; t >= 0; t--) states[t] = psi[t+1][states[t+1]];
  return Array.from(states);
}

function computeRegime(niftyBars) {
  if (!_hmmParams || !niftyBars || niftyBars.length < 10) {
    return { regime: _currentRegime || 'Unknown', state: -1, score: 0.5 };
  }

  const { scaler_mean, scaler_scale, regime_map } = _hmmParams;
  const bars   = niftyBars.slice(-250);
  const closes = bars.map(b => b.close);
  const vols   = bars.map(b => b.volume || 1);

  const k50 = 2 / 51, k200 = 2 / 201;
  let e50 = closes[0], e200 = closes[0];
  const e50s = [], e200s = [];
  for (const c of closes) {
    e50  = c * k50  + e50  * (1 - k50);
    e200 = c * k200 + e200 * (1 - k200);
    e50s.push(e50); e200s.push(e200);
  }

  const obsMatrix = [];
  for (let i = 20; i < bars.length; i++) {
    const ret      = closes[i] / closes[i-1] - 1;
    const retSlice = [];
    for (let k = Math.max(1, i - 19); k <= i; k++) retSlice.push(closes[k] / closes[k-1] - 1);
    const vol20      = stdDev(retSlice);
    const trend      = (e50s[i] - e200s[i]) / (e200s[i] || 1);
    const vol20avg   = mean(vols.slice(i-20, i));
    const vol_ratio  = vol20avg > 0 ? vols[i] / vol20avg : 1;
    const sgv        = sgVelocity(closes.slice(Math.max(0, i-10), i+1));
    const raw        = [ret, vol20, trend, vol_ratio, sgv];
    const zs         = raw.map((v, k) => scaler_scale[k] > 0
      ? (v - scaler_mean[k]) / scaler_scale[k] : 0);
    obsMatrix.push(zs);
  }

  if (obsMatrix.length === 0) {
    return { regime: _currentRegime || 'Unknown', state: -1, score: 0.5 };
  }

  const states    = viterbiDecode(obsMatrix);
  const lastState = states[states.length - 1];
  const regime    = regime_map[String(lastState)] || 'Unknown';
  _currentRegime      = regime;
  _currentRegimeState = lastState;

  const SCORE_MAP = { 'Bull-Trend': 1.0, 'Bear-Trend': 0.4, 'Chop': 0.2, 'High-Vol-Panic': 0.0 };
  return { regime, state: lastState, score: SCORE_MAP[regime] ?? 0.5 };
}

// ─── Conformal Prediction ─────────────────────────────────────────────────────
function getConfidenceInterval(rawScore, alpha = 0.10) {
  if (!_conformalScores) {
    return { lower: Math.max(0, rawScore - 0.15), upper: Math.min(1, rawScore + 0.15), q: 0.15 };
  }
  const scores = _conformalScores.ensemble || [];
  if (scores.length === 0) {
    return { lower: Math.max(0, rawScore - 0.15), upper: Math.min(1, rawScore + 0.15), q: 0.15 };
  }
  const idx = Math.ceil((1 - alpha) * scores.length) - 1;
  const q   = scores[Math.min(Math.max(idx, 0), scores.length - 1)];
  return { lower: Math.max(0, rawScore - q), upper: Math.min(1, rawScore + q), q };
}

// ─── SHAP Gate Weights ────────────────────────────────────────────────────────
function getGateWeights() {
  return _gateWeights ? (_gateWeights.gate_weights || {}) : {};
}

function applyGateWeights(gates) {
  const weights = getGateWeights();
  const result  = {};
  for (const [gate, passed] of Object.entries(gates)) {
    result[gate] = passed ? (weights[gate] || 1.0) : 0;
  }
  return result;
}

// ─── LGBM2c Ensemble Score (pure JS) ─────────────────────────────────────────
/**
 * getEnsembleScore(features) — LGBM2c global + regime-specific blend
 * Returns: {stackScore, xgbScore, lgbmScore, regimeScore, confLower, confUpper}
 * or null if models not loaded.
 */
async function getEnsembleScore(features) {
  if (!_lgbmGlobal) return null;

  const vec = buildFeatureVector(features);

  // Global model score
  const globalScore = predictOne(_lgbmGlobal, vec);

  // Regime-specific model score (fall back to global if unavailable)
  let regimeScore = globalScore;
  const stateIdx  = _currentRegimeState;
  if (stateIdx >= 0 && stateIdx <= 3) {
    try {
      const regimeModel = loadModel(LGBM_REGIME_PATHS[stateIdx]);
      regimeScore = predictOne(regimeModel, vec);
    } catch { /* file missing or corrupt — use global */ }
  }

  // 50/50 blend (matches predict_server.py)
  const stackScore = 0.5 * globalScore + 0.5 * regimeScore;
  const ci = getConfidenceInterval(stackScore);

  return {
    stackScore,
    xgbScore:   null,
    lgbmScore:  stackScore,
    regimeScore,
    softScore:  null,
    confLower:  ci.lower,
    confUpper:  ci.upper,
  };
}

// ─── PPO Position Size (pure JS) ─────────────────────────────────────────────
/**
 * getPositionSize(features, _regime) — PPO actor forward pass
 * Returns: number in {0.0, 0.25, 0.50, 0.75, 1.0}
 */
async function getPositionSize(features, _regime) {
  if (!fs.existsSync(PPO_WEIGHTS_PATH)) return 0.5;

  const vec = buildFeatureVector(features);

  // LGBM2c score feeds into PPO input[56]
  const lgbm2cScore = _lgbmGlobal ? predictOne(_lgbmGlobal, vec) : 0.5;

  // PPO takes first 56 base features
  const feat56 = vec.slice(0, 56);

  try {
    const result = ppoPredict(feat56, lgbm2cScore, PPO_WEIGHTS_PATH);
    return result.size;
  } catch {
    return 0.5;
  }
}

// ─── Regime gate ──────────────────────────────────────────────────────────────
function isRegimeAllowed(regime) {
  if (!regime || regime === 'Unknown') return true;
  return regime === 'Bull-Trend' || regime === 'Bear-Trend';
}

// ─── Math utilities ───────────────────────────────────────────────────────────
function mean(arr) {
  if (!arr || arr.length === 0) return 0;
  return arr.reduce((s, x) => s + x, 0) / arr.length;
}

function stdDev(arr) {
  if (!arr || arr.length < 2) return 0.01;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1)) + 1e-8;
}

function sgVelocity(closes) {
  const n = closes.length;
  if (n < 5) return 0;
  const xs = Array.from({length: n}, (_, i) => i);
  const xm = mean(xs), ym = mean(closes);
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - xm) * (closes[i] - ym);
    den += (xs[i] - xm) ** 2;
  }
  return den > 0 ? (num / den) : 0;
}

// ─── Public API ───────────────────────────────────────────────────────────────
module.exports = {
  init,
  computeRegime,
  getConfidenceInterval,
  getGateWeights,
  applyGateWeights,
  getEnsembleScore,
  getPositionSize,
  isRegimeAllowed,
  getCurrentRegime:      () => _currentRegime || 'Unknown',
  getCurrentRegimeState: () => _currentRegimeState,
  isServerAvailable:     () => true,   // always available — pure JS
};
