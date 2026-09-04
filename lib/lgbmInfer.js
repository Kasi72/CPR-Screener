'use strict';
/**
 * Pure-JS LightGBM inference.
 * Loads compact tree JSON exported by scripts/export_models_to_js.py.
 *
 * Tree format per tree:
 *   sf  – split_feature indices  (length = num_leaves - 1)
 *   th  – thresholds             (same length)
 *   lc  – left_child             (same length; negative = leaf, index = ~value)
 *   rc  – right_child            (same length)
 *   lv  – leaf_values            (length = num_leaves)
 *
 * Prediction:
 *   raw = sum of leaf values across all trees
 *   prob = sigmoid(raw)   for binary classification
 */

const fs   = require('fs');
const path = require('path');

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

/**
 * Walk one tree for one sample.
 * @param {Object} tree  - {sf, th, lc, rc, lv}
 * @param {Float32Array|number[]} feat  - feature vector
 * @returns {number} leaf value
 */
function walkTree(tree, feat) {
  const { sf, th, lc, rc, lv } = tree;
  let node = 0;
  while (node >= 0) {
    if (feat[sf[node]] <= th[node]) {
      node = lc[node];
    } else {
      node = rc[node];
    }
  }
  return lv[~node];  // ~node converts negative child ref to leaf index
}

/**
 * Predict probability for one feature vector.
 * @param {Object} model  - parsed model JSON (has .trees array)
 * @param {number[]} feat - feature vector (length must match model.num_features)
 * @returns {number} probability in [0, 1]
 */
function predictOne(model, feat) {
  let raw = 0;
  for (const tree of model.trees) raw += walkTree(tree, feat);
  return sigmoid(raw);
}

/**
 * Predict for a batch.
 * @param {Object} model
 * @param {number[][]} feats
 * @returns {number[]}
 */
function predictBatch(model, feats) {
  return feats.map(f => predictOne(model, f));
}

/**
 * Load a model from a JSON file path.
 * Caches in memory; safe to call multiple times.
 */
const _cache = new Map();
function loadModel(jsonPath) {
  const abs = path.resolve(jsonPath);
  if (_cache.has(abs)) return _cache.get(abs);
  const data = JSON.parse(fs.readFileSync(abs, 'utf-8'));
  _cache.set(abs, data);
  return data;
}

module.exports = { loadModel, predictOne, predictBatch, sigmoid };
