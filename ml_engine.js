'use strict';
/**
 * ml_engine.js — JavaScript ML inference layer
 *
 * Responsibilities:
 *   1. HMM Viterbi decoder (from hmm_params.json) — runs locally, no Python needed
 *   2. Conformal prediction interval (from conformal_scores.json)
 *   3. Ensemble score via predict_server /predict_ensemble
 *   4. Position size via predict_server /position_size (with PPO weights)
 *   5. SHAP gate weight loader (from shap_gate_weights.json)
 *
 * All JSON model files are read from ./models/
 * predict_server must be running on port 5001 for ensemble/position endpoints.
 */

const fs   = require('fs');
const path = require('path');
const http = require('http');

const MODELS_DIR     = path.join(__dirname, 'models');
const PREDICT_PORT   = 5001;
const PREDICT_HOST   = '127.0.0.1';

// ─── State cache ──────────────────────────────────────────────────────────────
let _hmmParams       = null;
let _conformalScores = null;
let _gateWeights     = null;
let _currentRegime   = null;   // cached from last computeRegime call
let _serverAvailable = null;   // null = unknown, true/false after first probe
let _serverProbeTime = 0;      // ms timestamp of last probe
const _PROBE_TTL_MS  = 60_000; // re-probe after 60 s so server restarts are detected

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
}

// ─── HMM Viterbi ─────────────────────────────────────────────────────────────
/**
 * Gaussian log-likelihood for a single state component.
 * p: {means:[...], covars:[...]}  obs: number[]
 */
function gaussianLogLikelihood(means, covars, obs) {
  let ll = 0;
  for (let i = 0; i < means.length; i++) {
    const d   = obs[i] - means[i];
    const var_ = covars[i];
    ll += -0.5 * (Math.log(2 * Math.PI * var_) + d * d / var_);
  }
  return ll;
}

/**
 * viterbiDecode(obsMatrix) — full Viterbi decoding over observation sequence
 * obsMatrix: number[][] — [T, nFeatures], already z-scored using HMM scaler
 * Returns: number[] — state sequence
 */
function viterbiDecode(obsMatrix) {
  if (!_hmmParams) return null;
  const { n_components, startprob, transmat, means, covars } = _hmmParams;
  const T = obsMatrix.length;
  const N = n_components;

  const delta  = Array.from({length: T}, () => new Float64Array(N));
  const psi    = Array.from({length: T}, () => new Int32Array(N));

  // Initialise
  for (let j = 0; j < N; j++) {
    delta[0][j] = Math.log(startprob[j] + 1e-300) +
                  gaussianLogLikelihood(means[j], covars[j], obsMatrix[0]);
  }

  // Recursion
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

  // Backtrack
  const states = new Int32Array(T);
  let best = -Infinity;
  for (let j = 0; j < N; j++) {
    if (delta[T-1][j] > best) { best = delta[T-1][j]; states[T-1] = j; }
  }
  for (let t = T-2; t >= 0; t--) {
    states[t] = psi[t+1][states[t+1]];
  }
  return Array.from(states);
}

/**
 * computeRegime(niftyBars) — derive regime from recent Nifty OHLCV bars
 * niftyBars: [{date, open, high, low, close, volume}, ...]  (at least 30)
 * Returns: {regime: string, state: number, score: number}
 */
function computeRegime(niftyBars) {
  if (!_hmmParams || !niftyBars || niftyBars.length < 10) {
    return { regime: _currentRegime || 'Unknown', state: -1, score: 0.5 };
  }

  const { scaler_mean, scaler_scale, regime_map } = _hmmParams;

  // Build observation matrix: [ret, vol20, trend, vol_ratio, sg_vel]
  const bars   = niftyBars.slice(-250);  // need 200+ bars for EMA200 convergence
  const closes = bars.map(b => b.close);
  const vols   = bars.map(b => b.volume || 1);

  // EMA50 and EMA200 for trend feature (must match Python data_utils.py training)
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
    // vol20 must be std of RETURNS (matching Python: pd.Series(ret).rolling(20).std())
    const retSlice = [];
    for (let k = Math.max(1, i - 19); k <= i; k++) retSlice.push(closes[k] / closes[k-1] - 1);
    const vol20    = stdDev(retSlice);
    const trend    = (e50s[i] - e200s[i]) / (e200s[i] || 1);
    const vol20avg = mean(vols.slice(i-20, i));
    const vol_ratio = vol20avg > 0 ? vols[i] / vol20avg : 1;
    const sgv      = sgVelocity(closes.slice(Math.max(0, i-10), i+1));
    // z-score
    const raw = [ret, vol20, trend, vol_ratio, sgv];
    const zs  = raw.map((v, k) => scaler_scale[k] > 0
      ? (v - scaler_mean[k]) / scaler_scale[k] : 0);
    obsMatrix.push(zs);
  }

  if (obsMatrix.length === 0) {
    return { regime: _currentRegime || 'Unknown', state: -1, score: 0.5 };
  }

  const states = viterbiDecode(obsMatrix);
  const lastState = states[states.length - 1];
  const regime    = regime_map[String(lastState)] || 'Unknown';
  _currentRegime  = regime;

  // Regime score: Bull=1.0, Bear=0.4, Chop=0.2, Panic=0.0
  const SCORE_MAP = { 'Bull-Trend': 1.0, 'Bear-Trend': 0.4, 'Chop': 0.2, 'High-Vol-Panic': 0.0 };
  const score = SCORE_MAP[regime] ?? 0.5;

  return { regime, state: lastState, score };
}

// ─── Conformal Prediction ─────────────────────────────────────────────────────
/**
 * getConfidenceInterval(rawScore, alpha=0.10)
 * Returns {lower, upper, q} — conformal prediction interval
 */
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
  return {
    lower: Math.max(0, rawScore - q),
    upper: Math.min(1, rawScore + q),
    q,
  };
}

// ─── SHAP Gate Weights ────────────────────────────────────────────────────────
function getGateWeights() {
  if (!_gateWeights) return {};
  return _gateWeights.gate_weights || {};
}

/**
 * applyGateWeights(gates) — multiply gate truth values by SHAP importance
 * gates: {gateId: boolean}
 * Returns: {gateId: weight}  — 0 if gate failed, weight if passed
 */
function applyGateWeights(gates) {
  const weights = getGateWeights();
  const result  = {};
  for (const [gate, passed] of Object.entries(gates)) {
    const w = weights[gate] || 1.0;
    result[gate] = passed ? w : 0;
  }
  return result;
}

// ─── HTTP call to predict_server ─────────────────────────────────────────────
function callServer(endpoint, body, method = 'POST') {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify(body);
    const headers = { 'Content-Type': 'application/json' };
    if (method !== 'GET') headers['Content-Length'] = Buffer.byteLength(payload);
    const req = http.request({
      host:    PREDICT_HOST,
      port:    PREDICT_PORT,
      path:    endpoint,
      method,
      headers,
    }, (res) => {
      let data = '';
      res.on('data', d => { data += d; });
      res.on('end', () => {
        try { resolve(JSON.parse(data)); }
        catch (e) { reject(new Error('Invalid JSON from predict_server')); }
      });
    });
    req.on('error', reject);
    // destroy emits 'error' which calls reject — avoid double-rejection by not calling reject here
    req.setTimeout(3000, () => { req.destroy(new Error('predict_server timeout')); });
    if (method !== 'GET') req.write(payload);
    req.end();
  });
}

async function probeServer() {
  const now = Date.now();
  if (_serverAvailable !== null && (now - _serverProbeTime) < _PROBE_TTL_MS) {
    return _serverAvailable;
  }
  try {
    const r = await callServer('/health', {}, 'GET');
    _serverAvailable = r.status === 'ok';
  } catch { _serverAvailable = false; }
  _serverProbeTime = Date.now();
  return _serverAvailable;
}

// ─── Ensemble Score ───────────────────────────────────────────────────────────
/**
 * getEnsembleScore(features) — call predict_server for ensemble prediction
 * features: object with FEATURE_COLS keys
 * Returns: {stackScore, xgbScore, lgbmScore, confLower, confUpper} or null
 */
async function getEnsembleScore(features) {
  const available = await probeServer();
  if (!available) return null;
  try {
    const res = await callServer('/predict_ensemble', features);
    const e   = (res.ensemble || [])[0];
    if (!e) return null;
    return {
      stackScore:   e.stack_score,
      xgbScore:     e.xgb_score,
      lgbmScore:    e.lgbm_score,
      regimeScore:  e.regime_score,
      softScore:    e.soft_score,
      confLower:    e.conf_lower,
      confUpper:    e.conf_upper,
    };
  } catch {
    return null;
  }
}

// ─── Position Size ────────────────────────────────────────────────────────────
/**
 * getPositionSize(features, regime) — call predict_server for PPO sizing
 * Returns: number (0, 0.25, 0.50, 0.75, or 1.00) or 0.5 as default
 */
async function getPositionSize(features, regime) {
  const available = await probeServer();
  if (!available) return 0.5;
  try {
    const res = await callServer(
      `/position_size?regime=${encodeURIComponent(regime || 'Bull-Trend')}`,
      features
    );
    const sizes = res.position_sizes || [];
    return sizes[0] ?? 0.5;
  } catch {
    return 0.5;
  }
}

// ─── Regime-based signal gate ─────────────────────────────────────────────────
/**
 * isRegimeAllowed(regime) — block Chop / High-Vol-Panic from producing signals
 */
function isRegimeAllowed(regime) {
  if (!regime || regime === 'Unknown') return true;   // no model = allow
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

function sgVelocity(closes, deg = 2) {
  const n = closes.length;
  if (n < 5) return 0;
  // simple linear regression slope as SG-velocity proxy
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
  getCurrentRegime: () => _currentRegime || 'Unknown',
  isServerAvailable: () => _serverAvailable,
};
