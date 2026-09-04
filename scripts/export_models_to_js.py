"""
Export LGBM2c models and PPO weights to compact JS-friendly JSON.

Outputs:
  models/lgbm2c_global_js.json
  models/lgbm2c_regime_js.json   (dict keyed by regime int 0-3)
  models/ppo_policy_weights.json (already exists, verify only)

Run once after any retrain:
  python scripts/export_models_to_js.py
"""

import json, os, re, struct, sys
import numpy as np

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE, 'models')


# ── LightGBM text-format parser ───────────────────────────────────────────────
# Parses the .txt format LightGBM saves natively — no lgb import needed.

def _ints(s):   return [int(x)   for x in s.strip().split() if x]
def _floats(s): return [float(x) for x in s.strip().split() if x]

def parse_lgbm_txt(path):
    """Parse LightGBM model .txt file into compact JS tree format."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Extract feature names (LightGBM writes: feature_names=f1 f2 ...)
    m = re.search(r'^feature_names=(.+)$', text, re.M)
    feature_names = m.group(1).strip().split() if m else []

    # Extract objective to confirm binary
    m = re.search(r'^objective=(.+)$', text, re.M)
    objective = m.group(1).strip() if m else ''

    # Split into per-tree blocks
    tree_blocks = re.split(r'\nTree=\d+\n', text)
    tree_blocks = tree_blocks[1:]   # drop header before first Tree=

    trees = []
    for block in tree_blocks:
        def _get(key):
            m = re.search(rf'^{key}=(.+)$', block, re.M)
            return m.group(1) if m else ''

        num_leaves = int(_get('num_leaves') or 0)
        if num_leaves == 0:
            continue

        num_internals = num_leaves - 1

        sf  = _ints(_get('split_feature'))[:num_internals]
        th  = []
        raw_th = _get('threshold').strip().split()[:num_internals]
        for t in raw_th:
            try:    th.append(float(t))
            except: th.append(0.0)

        lc  = _ints(_get('left_child'))[:num_internals]
        rc  = _ints(_get('right_child'))[:num_internals]
        lv  = _floats(_get('leaf_value'))[:num_leaves]

        # Compact: round to 6dp to save JSON bytes
        th = [round(v, 6) for v in th]
        lv = [round(v, 8) for v in lv]

        trees.append({'sf': sf, 'th': th, 'lc': lc, 'rc': rc, 'lv': lv})

    return {
        'num_trees':     len(trees),
        'num_features':  len(feature_names),
        'feature_names': feature_names,
        'objective':     objective,
        'trees':         trees,
    }


def export_lgbm(name, src_path, dst_path):
    print(f'Parsing {name} ({os.path.getsize(src_path)//1024}KB)…')
    data = parse_lgbm_txt(src_path)
    with open(dst_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, separators=(',', ':'))
    print(f'  -> {dst_path} ({os.path.getsize(dst_path)//1024}KB, {data["num_trees"]} trees)')
    return data


# ── Export global LGBM2c ──────────────────────────────────────────────────────
global_src = os.path.join(MODELS_DIR, 'lgbm2c_global.txt')
global_dst = os.path.join(MODELS_DIR, 'lgbm2c_global_js.json')
export_lgbm('lgbm2c_global', global_src, global_dst)


# ── Export regime LGBM2c models ───────────────────────────────────────────────
regime_out = {}
for i in range(4):
    src = os.path.join(MODELS_DIR, f'lgbm2c_regime_{i}.txt')
    if not os.path.exists(src):
        print(f'  lgbm2c_regime_{i}.txt missing — skip')
        continue
    dst = os.path.join(MODELS_DIR, f'lgbm2c_regime_{i}_js.json')
    data = export_lgbm(f'lgbm2c_regime_{i}', src, dst)
    regime_out[str(i)] = dst

# Write manifest of regime model paths
manifest = {'regime_model_files': regime_out}
manifest_path = os.path.join(MODELS_DIR, 'lgbm2c_regime_manifest.json')
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)
print(f'\nManifest: {manifest_path}')


# ── Verify PPO weights ────────────────────────────────────────────────────────
ppo_path = os.path.join(MODELS_DIR, 'ppo_policy_weights.json')
if os.path.exists(ppo_path):
    with open(ppo_path) as f:
        ppo = json.load(f)
    keys = list(ppo.keys())
    w0 = np.array(ppo['mlp_extractor.policy_net.0.weight'])
    w2 = np.array(ppo['mlp_extractor.policy_net.2.weight'])
    wa = np.array(ppo['action_net.weight'])
    print(f'\nPPO weights OK: {w0.shape} -> {w2.shape} -> {wa.shape}')
    print(f'  Input dim={w0.shape[1]}, Action dim={wa.shape[0]}')
else:
    print('\nWARN: ppo_policy_weights.json not found')

print('\nDone. All models ready for JS inference.')
