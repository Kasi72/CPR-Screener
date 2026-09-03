"""
score_p2c.py — Populate lgbm2c_score in signal_dataset.csv using local Phase 2c models.
Computes the 5 interaction features, applies direction-adjustment, scores with
lgbm2c_global.txt, then saves back to the CSV.

Usage:
    python scripts/ml/score_p2c.py
"""

import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

BASE       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE, 'models')
SIGNAL_CSV = os.path.join(MODELS_DIR, 'signal_dataset.csv')

BASE_FEATURES = [
    'cpr_width_pct', 'vwap_dist', 'atr_pct_rank', 'vol_rank',
    'n_rules_fired', 'sg_vel', 'ema200_dist', 'rsi14',
    'mom5', 'dow', 'rule_id', 'direction',
    'dist_hi52', 'dist_lo52', 'vol_accel',
    'market_rs_5d', 'market_rs_20d', 'sector_rs_5d', 'sector_rs_20d',
    'deliv_pct', 'pcr', 'india_vix',
    'conf_vol', 'rsi_dir', 'hi52_dir',
    'cpr_compress', 'cpr_pos', 'dist_r1', 'dist_s1',
    'mom3', 'mom10', 'mom20',
    'rsi_div', 'vol_accel_delta',
    'days_since_52hi', 'expiry_dist',
    'cpr_overlap_pct', 'open_to_cpr_dist', 'prev_cpr_respected', 'cpr_zone_vol_ratio',
    'hmm_regime',
    # Sprint 2A: compression/structure CPR features
    'open_inside_cpr', 'cpr_virgin', 'consecutive_narrow_cprs',
    'cpr_midpoint_trend', 'cpr_expansion_factor',
    # Sprint 2B: structural + context CPR features
    'cpr_above_prev_cpr', 'prev_close_inside_cpr', 'atr_to_cpr_ratio',
    'cpr_width_percentile_252d', 'prev_day_ochoa_type',
    # Sprint 3: gap + bar quality + volatility + volume structure
    'gap_pct', 'cpr_test_count_5d', 'prev_bar_close_pos',
    'atr_expansion', 'vol_trend_slope',
]  # 56

INTERACTION_FEATURES = [
    'cpr_vol_interaction',
    'regime_momentum',
    'cpr_rsi_squeeze',
    'overlap_vol_signal',
    'rs_direction_alignment',
    'virgin_momentum',
    'narrow_breakout_vol',
]  # 7

FEATURE_COLS = BASE_FEATURES + INTERACTION_FEATURES  # 58 total (51 base + 7 interactions)

DIRECTIONAL_FEATURES = {
    'dist_hi52', 'dist_lo52', 'vwap_dist', 'ema200_dist',
    'mom3', 'mom5', 'mom10', 'mom20',
    'market_rs_5d', 'market_rs_20d', 'sector_rs_5d', 'sector_rs_20d',
    'cpr_pos', 'dist_r1', 'dist_s1', 'sg_vel',
    'open_to_cpr_dist',
    'gap_pct',          # gap-up = good for long, gap-down = good for short
    # prev_bar_close_pos excluded: semantics ambiguous for shorts (near-high = resistance)
}


def compute_interactions(df):
    df = df.copy()
    df['cpr_vol_interaction']    = df['cpr_compress'] * df['vol_rank']
    df['regime_momentum']        = df['hmm_regime']   * df['mom5']
    df['cpr_rsi_squeeze']        = (1 - df['cpr_width_pct']) * df['rsi14']
    df['overlap_vol_signal']     = df['cpr_overlap_pct'] * df['cpr_zone_vol_ratio']
    df['rs_direction_alignment'] = (df['market_rs_5d'] + df['sector_rs_5d']) * df['direction']
    df['virgin_momentum']        = df['cpr_virgin'] * df['mom5']
    df['narrow_breakout_vol']    = df['consecutive_narrow_cprs'] * df['vol_rank']
    return df


def apply_direction_adjustment(X, directions):
    X = X.copy()
    dir_cols = [c for c in DIRECTIONAL_FEATURES if c in X.columns]
    sell_mask = (directions == -1).values
    X.loc[sell_mask, dir_cols] = X.loc[sell_mask, dir_cols] * -1
    return X


def main():
    print(f"Loading {SIGNAL_CSV} ...")
    df = pd.read_csv(SIGNAL_CSV, low_memory=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # Check existing coverage
    existing_coverage = df['lgbm2c_score'].notna().mean() if 'lgbm2c_score' in df.columns else 0.0
    print(f"  Existing lgbm2c_score coverage: {existing_coverage:.1%}")

    # Compute interaction features
    print("  Computing 5 interaction features ...")
    df = compute_interactions(df)

    # Validate all features present
    missing = [f for f in FEATURE_COLS if f not in df.columns]
    if missing:
        print(f"  ERROR: Missing features: {missing}")
        sys.exit(1)

    # Load Phase 2c global model
    global_path = os.path.join(MODELS_DIR, 'lgbm2c_global.txt')
    if not os.path.exists(global_path):
        print(f"  ERROR: {global_path} not found")
        sys.exit(1)
    print(f"  Loading {global_path} ...")
    global_model = lgb.Booster(model_file=global_path)
    print(f"    Model expects {global_model.num_feature()} features")

    # Apply direction adjustment
    X = df[FEATURE_COLS].copy()
    X = apply_direction_adjustment(X, df['direction'])
    # Encode any remaining string columns as category codes
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category').cat.codes
    X = X.fillna(0).astype(np.float32)

    # Score with global model
    print("  Scoring with lgbm2c_global.txt ...")
    scores = global_model.predict(X.values)

    # Try regime-specific models and blend
    regime_scores = np.zeros(len(df))
    regime_counts  = np.zeros(len(df))
    for regime in range(4):
        rpath = os.path.join(MODELS_DIR, f'lgbm2c_regime_{regime}.txt')
        if not os.path.exists(rpath):
            continue
        rm = lgb.Booster(model_file=rpath)
        if rm.num_feature() != len(FEATURE_COLS):
            print(f"    Regime {regime}: feature mismatch ({rm.num_feature()} vs {len(FEATURE_COLS)}) — skip")
            continue
        mask = (df['hmm_regime'] == regime).values
        if mask.sum() == 0:
            continue
        regime_scores[mask] = rm.predict(X.values[mask])
        regime_counts[mask] = 1
        print(f"    Regime {regime}: {mask.sum():,} rows scored")

    # Blend: 50% global + 50% regime (where regime model exists)
    final_scores = scores.copy()
    has_regime = regime_counts > 0
    final_scores[has_regime] = 0.5 * scores[has_regime] + 0.5 * regime_scores[has_regime]

    df['lgbm2c_score'] = final_scores
    print(f"  lgbm2c_score: min={final_scores.min():.4f}  max={final_scores.max():.4f}  mean={final_scores.mean():.4f}")

    # Save back
    print(f"  Saving to {SIGNAL_CSV} ...")
    df.to_csv(SIGNAL_CSV, index=False)
    print(f"  Done. lgbm2c_score coverage: {df['lgbm2c_score'].notna().mean():.1%}")


if __name__ == '__main__':
    main()
