'use strict';
/**
 * Pure-JS PPO actor inference.
 * Architecture: input(60) → Linear(128) → Tanh → Linear(64) → Tanh → Linear(5) → Softmax
 *
 * Input: 56 base features + [lgbm2c_score, exposure=0, cum_pnl=0, win_streak=0]
 * Output: argmax over 5 actions = position size index
 *   0=Skip, 1=Quarter, 2=Half, 3=Three-Quarter, 4=Full
 */

const fs   = require('fs');
const path = require('path');

const POSITION_LABELS = ['Skip', 'Quarter', 'Half', 'Three-Quarter', 'Full'];

/** Matrix-vector multiply: W[rows×cols] · x[cols] + b[rows] → out[rows] */
function linear(W, b, x) {
  if (!W || !W.length || !W[0]) return new Float32Array(0);
  const rows = W.length;
  const cols = W[0].length;
  const out  = new Float32Array(rows);
  for (let i = 0; i < rows; i++) {
    let s = b[i];
    const wi = W[i];
    for (let j = 0; j < cols; j++) s += wi[j] * x[j];
    out[i] = s;
  }
  return out;
}

function tanh(arr) {
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = Math.tanh(arr[i]);
  return out;
}

function softmax(arr) {
  let max = -Infinity;
  for (const v of arr) if (v > max) max = v;
  let sum = 0;
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) { out[i] = Math.exp(arr[i] - max); sum += out[i]; }
  for (let i = 0; i < arr.length; i++) out[i] /= sum;
  return out;
}

let _weights = null;

function loadWeights(jsonPath) {
  if (_weights) return _weights;
  let raw;
  try { raw = JSON.parse(fs.readFileSync(path.resolve(jsonPath), 'utf-8')); } catch (e) { throw new Error(`PPO weights parse: ${e.message}`); }
  const KEYS = ['mlp_extractor.policy_net.0.weight', 'mlp_extractor.policy_net.0.bias',
                'mlp_extractor.policy_net.2.weight', 'mlp_extractor.policy_net.2.bias',
                'action_net.weight', 'action_net.bias'];
  const missing = KEYS.filter(k => !(k in raw));
  if (missing.length) throw new Error(`PPO weights missing keys: ${missing.join(', ')}`);
  _weights = {
    w0: raw['mlp_extractor.policy_net.0.weight'],
    b0: raw['mlp_extractor.policy_net.0.bias'],
    w2: raw['mlp_extractor.policy_net.2.weight'],
    b2: raw['mlp_extractor.policy_net.2.bias'],
    wa: raw['action_net.weight'],
    ba: raw['action_net.bias'],
  };
  return _weights;
}

/**
 * Run PPO actor forward pass.
 * @param {number[]} feat56  - 56 base P2C features (FEATURE_COLS order)
 * @param {number}   lgbm2cScore  - LGBM2c probability [0,1]
 * @returns {{ label: string, probs: number[], action: number }}
 */
function ppoPredict(feat56, lgbm2cScore, weightsPath) {
  const w = loadWeights(weightsPath);

  // Build 60-dim input: 56 features + lgbm2c_score + exposure=0 + cum_pnl=0 + win_streak=0
  const x = new Float32Array(60);
  for (let i = 0; i < 56; i++) x[i] = feat56[i] || 0;
  x[56] = lgbm2cScore || 0;
  // x[57], x[58], x[59] = 0 (exposure, cum_pnl, win_streak — not available at scan time)

  const h0     = tanh(linear(w.w0, w.b0, x));
  const h1     = tanh(linear(w.w2, w.b2, h0));
  const logits = linear(w.wa, w.ba, h1);
  const probs  = softmax(logits);

  let action = 0;
  let best   = probs[0];
  for (let i = 1; i < probs.length; i++) if (probs[i] > best) { best = probs[i]; action = i; }

  return {
    action,
    label:  POSITION_LABELS[action],
    size:   action / 4,          // 0.0 – 1.0 continuous
    probs:  Array.from(probs),
  };
}

module.exports = { loadWeights, ppoPredict, POSITION_LABELS };
