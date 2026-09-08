"""
data_utils.py — Shared constants, TA indicators, and feature engineering.

Sections
--------
1. CONSTANTS    — paths, feature lists, constraints (imported by all phases)
2. TA HELPERS   — ema, rsi, atr, cpr, vwap (pure functions, no side effects)
3. FEATURE API  — build_features(), build_nifty_regime_features()
4. DATA LOADER  — load_signal_dataset()
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ── 1. CONSTANTS ──────────────────────────────────────────────────────────────

import os as _os
_HERE      = _os.path.dirname(_os.path.abspath(__file__))
_BASE      = _os.path.dirname(_os.path.dirname(_HERE))
DATA_FILE  = _os.environ.get('NSE_DATA_FILE', r"C:\Users\drkkr\Downloads\NIFTY 500 OHLCV\ALL_SYMBOLS_OHLCV.csv")
MODELS_DIR = _os.environ.get('NSE_MODELS_DIR', _os.path.join(_BASE, 'models'))

FEATURE_COLS = [
    # original 12
    'cpr_width_pct', 'vwap_dist', 'atr_pct_rank', 'vol_rank',
    'n_rules_fired', 'sg_vel', 'ema200_dist', 'rsi14',
    'mom5', 'dow', 'rule_id', 'direction',
    # Phase A: 52-week context + volume acceleration
    'dist_hi52', 'dist_lo52', 'vol_accel',
    # Phase B: relative strength (market + sector)
    'market_rs_5d', 'market_rs_20d', 'sector_rs_5d', 'sector_rs_20d',
    # Phase C: delivery % (smart money proxy)
    'deliv_pct',
    # Phase D: options put-call ratio (institutional hedging signal)
    'pcr',
    # Tier 1: VIX + interaction features
    'india_vix',         # market fear level
    'conf_vol',          # n_rules_fired × vol_accel
    'rsi_dir',           # rsi14 × direction
    'hi52_dir',          # dist_hi52 × direction
    # Tier 2A: CPR quality
    'cpr_compress',      # today CPR width / 5d avg
    'cpr_pos',           # (close - cpr_lower) / cpr_width clipped [0,1]
    'dist_r1',           # (close - R1) / close
    'dist_s1',           # (close - S1) / close
    # Tier 2B: multi-timeframe momentum
    'mom3', 'mom10', 'mom20',
    # Tier 2C: divergence + volume curvature
    'rsi_div', 'vol_accel_delta',
    # Tier 2D: context
    'days_since_52hi', 'expiry_dist',
    # Sprint 1: CPR zone quality features
    'cpr_overlap_pct',       # daily CPR overlap with prior CPR (support zone quality)
    'open_to_cpr_dist',      # open vs CPR pivot distance / ATR (entry proximity)
    'prev_cpr_respected',    # 1 if prior day CPR held as support/resistance
    'cpr_zone_vol_ratio',    # vol in CPR band vs total vol (zone conviction)
    'hmm_regime',            # HMM market regime 0-3 (-1 if absent)
    # Sprint 2A: CPR compression + structure
    'open_inside_cpr',           # open printed inside CPR range
    'cpr_virgin',                # CPR not yet tested today (first break = strongest)
    'consecutive_narrow_cprs',   # streak of progressively narrower CPRs (squeeze)
    'cpr_midpoint_trend',        # slope of 5d CPR midpoint (trending structure)
    'cpr_expansion_factor',      # today CPR width / prior CPR width
    # Sprint 2B: structural + context CPR features
    'cpr_above_prev_cpr',        # today CPR above yesterday's (bullish structure)
    'prev_close_inside_cpr',     # prev close inside CPR (indecision carry-through)
    'atr_to_cpr_ratio',          # ATR / CPR width (breakout range potential)
    'cpr_width_percentile_252d', # CPR width rank over 252d (relative compression)
    'prev_day_ochoa_type',       # yesterday's OCHOA candle type
    # Sprint 3: gap + bar quality + volatility + volume structure
    'gap_pct',            # today open vs prev close / prev close
    'cpr_test_count_5d',  # how many last 5 bars tested CPR
    'prev_bar_close_pos', # yesterday close position in H-L range [0=low, 1=high]
    'atr_expansion',      # ATR today / 5d avg ATR
    'vol_trend_slope',    # 5-day volume slope / avg
    # Sprint 4: weekly CPR features
    'weekly_cpr_first_break',  # first break of weekly CPR this week
    'weekly_price_above_wtc',  # price above weekly top-of-CPR (TC)
]  # 58 — matches Phase 2c BASE_FEATURES and Phase 4 FEATURE_COLS exactly

# Monotone constraints aligned with FEATURE_COLS (58 features).
# +1 = feature↑ → win rate↑  |  -1 = feature↑ → win rate↓  |  0 = no constraint
MONOTONE_CONSTRAINTS = [
    0,   # cpr_width_pct
    0,   # vwap_dist
    0,   # atr_pct_rank
    1,   # vol_rank: higher vol rank → better breakout confirmation
    1,   # n_rules_fired: more confluence → better
    0,   # sg_vel
    0,   # ema200_dist
    0,   # rsi14 (directional effect captured by rsi_dir)
    0,   # mom5
    0,   # dow
    0,   # rule_id
    0,   # direction
    0,   # dist_hi52 (captured by hi52_dir)
    0,   # dist_lo52
    1,   # vol_accel: volume surge → better confirmation
    0,   # market_rs_5d
    0,   # market_rs_20d
    0,   # sector_rs_5d
    0,   # sector_rs_20d
    1,   # deliv_pct: higher delivery → institutional conviction
    0,   # pcr
   -1,   # india_vix: higher fear → worse setup quality
    1,   # conf_vol: confluence × volume → better
   -1,   # rsi_dir: overbought long / oversold short → worse
    0,   # hi52_dir
    0,   # cpr_compress: nonlinear, no monotone direction
    0,   # cpr_pos
    0,   # dist_r1
    0,   # dist_s1
    0,   # mom3
    0,   # mom10
    0,   # mom20
    0,   # rsi_div
    1,   # vol_accel_delta: accelerating volume surge → better
    0,   # days_since_52hi
    0,   # expiry_dist
    # Sprint 1
    1,   # cpr_overlap_pct: more overlap = stronger CPR zone = better signal quality
    0,   # open_to_cpr_dist: direction-dependent, model learns
    1,   # prev_cpr_respected: confirmed zone → better setup
    1,   # cpr_zone_vol_ratio: more zone vol = stronger conviction
    0,   # hmm_regime: 0-3 ordinal, not monotone
    # Sprint 2A
    0,   # open_inside_cpr: ambiguous (inside = indecision)
    1,   # cpr_virgin: untested CPR → stronger first-break setup
    1,   # consecutive_narrow_cprs: tighter squeeze → bigger breakout potential
    0,   # cpr_midpoint_trend: direction-dependent
    0,   # cpr_expansion_factor: nonlinear effect
    # Sprint 2B
    1,   # cpr_above_prev_cpr: bullish structure for longs
    0,   # prev_close_inside_cpr: indecision signal, ambiguous
    0,   # atr_to_cpr_ratio: high = more range but also more risk
    0,   # cpr_width_percentile_252d: context-dependent
    0,   # prev_day_ochoa_type: categorical, nonlinear
    # Sprint 3
    0,   # gap_pct: direction-dependent (gap-up good for long, bad for short)
    1,   # cpr_test_count_5d: more tests → stronger zone
    1,   # prev_bar_close_pos: closed near high → bullish carry-through
    0,   # atr_expansion: expanding = more range, model learns direction
    1,   # vol_trend_slope: volume building = breakout confirmation
    # Sprint 4
    1,   # weekly_cpr_first_break: momentum signal
    1,   # weekly_price_above_wtc: bullish weekly structure
]  # 58 — must match FEATURE_COLS length

SEQUENCE_COLS = ['ret', 'hl_range', 'vol_ratio', 'rsi14', 'sg_vel', 'mom5']

# Target column — 'hit_t1' is recommended (full T1 hit).
# Alternatives: 'win' (legacy 0.5% threshold), 'win_rr' (>1.5x ATR%), 'rr_ratio' (continuous).
WIN_COL = 'hit_t1'


# ── 2. TA HELPERS ─────────────────────────────────────────────────────────────

def ema(arr, span):
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values


def rsi_wilder(closes, period=14):
    c = np.array(closes, dtype=float)
    if len(c) < period + 2:
        return 50.0
    d = np.diff(c)
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    ag = g[:period].mean()
    al = l[:period].mean()
    for j in range(period, len(d)):
        ag = (ag * (period - 1) + g[j]) / period
        al = (al * (period - 1) + l[j]) / period
    return 100.0 - (100.0 / (1 + ag / al)) if al > 0 else 100.0


def sg_vel(closes, window=11, poly=3):
    if len(closes) < window:
        return 0.0
    smoothed = savgol_filter(closes, window, poly, deriv=1)
    return float(smoothed[-1])


def atr(highs, lows, closes, period=14):
    h, l, c = np.array(highs), np.array(lows), np.array(closes)
    tr = np.maximum(h[1:] - l[1:], np.maximum(
        np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    if len(tr) < period:
        return tr.mean() if len(tr) > 0 else 0.0
    atr_val = tr[:period].mean()
    for v in tr[period:]:
        atr_val = (atr_val * (period - 1) + v) / period
    return atr_val


def atr_pct_rank(highs, lows, closes, period=14, window=252):
    atrs = []
    for i in range(period, len(closes)):
        atrs.append(atr(highs[max(0, i-window):i],
                        lows[max(0, i-window):i],
                        closes[max(0, i-window):i], period))
    if not atrs:
        return 0.5
    cur = atrs[-1]
    return float(np.mean(np.array(atrs) <= cur))


def calc_cpr(H, L, C):
    pivot = (H + L + C) / 3
    bc = (H + L) / 2
    tc = 2 * pivot - bc
    upper, lower = max(tc, bc), min(tc, bc)
    w = upper - lower
    wp = (w / pivot * 100) if pivot > 0 else 0
    r1 = 2 * pivot - L   # classic floor pivot R1
    s1 = 2 * pivot - H   # classic floor pivot S1
    return dict(pivot=pivot, upper=upper, lower=lower, tc=tc, bc=bc,
                width=w, width_pct=wp, r1=r1, s1=s1)


def expiry_dist_days(dt):
    """Days from dt to the next monthly F&O expiry (last Thursday of month)."""
    import calendar
    dt = pd.Timestamp(dt)
    for month_offset in range(0, 3):
        m = (dt.month - 1 + month_offset) % 12 + 1
        y = dt.year + (dt.month - 1 + month_offset) // 12
        last_day = calendar.monthrange(y, m)[1]
        exp = pd.Timestamp(y, m, last_day)
        days_back = (exp.weekday() - 3) % 7  # Thursday = weekday 3
        exp = exp - pd.Timedelta(days=days_back)
        if exp >= dt:
            return int((exp - dt).days)
    return 30


def calc_vwap(opens, highs, lows, closes, volumes):
    tp = (highs + lows + closes) / 3
    pv = np.cumsum(tp * volumes)
    cv = np.cumsum(volumes)
    return np.where(cv > 0, pv / cv, tp)


# ── 3. FEATURE API ────────────────────────────────────────────────────────────

def build_features(row: dict, rule_map: Optional[dict] = None) -> dict:
    """Convert a signal row dict to the 58-feature vector dict (FEATURE_COLS order)."""
    if rule_map is None:
        rule_map = {f'rule{i}': i for i in range(1, 17)}  # rules 1–16
    return {
        # original 12
        'cpr_width_pct': float(row.get('cpr_width_pct', 0)),
        'vwap_dist':     float(row.get('vwap_dist', 0)),
        'atr_pct_rank':  float(row.get('atr_pct_rank', 0.5)),
        'vol_rank':      float(row.get('vol_rank', 0.5)),
        'n_rules_fired': float(row.get('n_rules_fired', 1)),
        'sg_vel':        float(row.get('sg_vel', 0)),
        'ema200_dist':   float(row.get('ema200_dist', 0)),
        'rsi14':         float(row.get('rsi14', 50)),
        'mom5':          float(row.get('mom5', 0)),
        'dow':           float(row.get('dow', 2)),
        'rule_id':       float(rule_map.get(row.get('rule_id', 'rule1'), 1)),
        'direction':     float(row.get('direction', 1)),
        # Phase A: 52w + vol accel
        'dist_hi52':     float(row.get('dist_hi52', -0.1)),
        'dist_lo52':     float(row.get('dist_lo52',  0.1)),
        'vol_accel':     float(row.get('vol_accel',  1.0)),
        # Phase B: RS
        'market_rs_5d':  float(row.get('market_rs_5d',  1.0)),
        'market_rs_20d': float(row.get('market_rs_20d', 1.0)),
        'sector_rs_5d':  float(row.get('sector_rs_5d',  1.0)),
        'sector_rs_20d': float(row.get('sector_rs_20d', 1.0)),
        # Phase C: delivery
        'deliv_pct':     float(row.get('deliv_pct', 0.0)),
        # Phase D: PCR
        'pcr':           float(row.get('pcr', 1.0)),
        # Tier 1: VIX + interactions
        'india_vix':     float(row.get('india_vix', 15.0)),
        'conf_vol':      float(row.get('conf_vol', 2.0)),
        'rsi_dir':       float(row.get('rsi_dir', 0.0)),
        'hi52_dir':      float(row.get('hi52_dir', 0.0)),
        # Tier 2A: CPR quality
        'cpr_compress':  float(row.get('cpr_compress', 1.0)),
        'cpr_pos':       float(row.get('cpr_pos', 0.5)),
        'dist_r1':       float(row.get('dist_r1', -0.02)),
        'dist_s1':       float(row.get('dist_s1',  0.02)),
        # Tier 2B: multi-timeframe momentum
        'mom3':          float(row.get('mom3', 0.0)),
        'mom10':         float(row.get('mom10', 0.0)),
        'mom20':         float(row.get('mom20', 0.0)),
        # Tier 2C: divergence + volume curvature
        'rsi_div':       float(row.get('rsi_div', 0.0)),
        'vol_accel_delta': float(row.get('vol_accel_delta', 0.0)),
        # Tier 2D: context
        'days_since_52hi': float(row.get('days_since_52hi', 90.0)),
        'expiry_dist':   float(row.get('expiry_dist', 15.0)),
        # Sprint 1: CPR zone quality
        'cpr_overlap_pct':    float(row.get('cpr_overlap_pct', 0.0)),
        'open_to_cpr_dist':   float(row.get('open_to_cpr_dist', 0.0)),
        'prev_cpr_respected': float(row.get('prev_cpr_respected', 0.0)),
        'cpr_zone_vol_ratio': float(row.get('cpr_zone_vol_ratio', 1.0)),
        'hmm_regime':         float(row.get('hmm_regime', -1)),
        # Sprint 2A: CPR compression + structure
        'open_inside_cpr':          float(row.get('open_inside_cpr', 0.0)),
        'cpr_virgin':               float(row.get('cpr_virgin', 1.0)),
        'consecutive_narrow_cprs':  float(row.get('consecutive_narrow_cprs', 0.0)),
        'cpr_midpoint_trend':       float(row.get('cpr_midpoint_trend', 0.0)),
        'cpr_expansion_factor':     float(row.get('cpr_expansion_factor', 1.0)),
        # Sprint 2B: structural + context CPR features
        'cpr_above_prev_cpr':        float(row.get('cpr_above_prev_cpr', 0.0)),
        'prev_close_inside_cpr':     float(row.get('prev_close_inside_cpr', 0.0)),
        'atr_to_cpr_ratio':          float(row.get('atr_to_cpr_ratio', 1.0)),
        'cpr_width_percentile_252d': float(row.get('cpr_width_percentile_252d', 0.5)),
        'prev_day_ochoa_type':       float(row.get('prev_day_ochoa_type', 0.0)),
        # Sprint 3: gap + bar quality + volatility + volume structure
        'gap_pct':            float(row.get('gap_pct', 0.0)),
        'cpr_test_count_5d':  float(row.get('cpr_test_count_5d', 0)),
        'prev_bar_close_pos': float(row.get('prev_bar_close_pos', 0.5)),
        'atr_expansion':      float(row.get('atr_expansion', 1.0)),
        'vol_trend_slope':    float(row.get('vol_trend_slope', 0.0)),
        # Sprint 4: weekly CPR features
        'weekly_cpr_first_break': float(row.get('weekly_cpr_first_break', 0.0)),
        'weekly_price_above_wtc': float(row.get('weekly_price_above_wtc', 0.0)),
    }


# -- Nifty 50 regime features -------------------------------------------------

def build_nifty_regime_features(nifty_df):
    """
    Build HMM observation sequence from Nifty 50 daily OHLCV.
    Returns numpy array [T, 5].
    """
    df = nifty_df.copy().sort_index()
    # Flatten MultiIndex columns from yfinance (e.g. ('Close','^NSEI') ??? 'Close')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    closes  = df['Close'].values.flatten()
    highs   = df['High'].values.flatten()
    lows    = df['Low'].values.flatten()
    vol_col = df['Volume'] if 'Volume' in df.columns else pd.Series(np.ones(len(df)), index=df.index)
    volumes = vol_col.values.flatten()

    ret      = np.diff(closes) / closes[:-1]               # daily return
    vol20    = pd.Series(ret).rolling(20).std().values      # 20-day vol
    ema50_   = ema(closes[1:], 50)
    ema200_  = ema(closes[1:], 200)
    trend    = (ema50_ - ema200_) / np.where(ema200_ > 0, ema200_, 1)

    vol20avg = pd.Series(volumes[1:]).rolling(20).mean().values
    vol_r    = np.where(vol20avg > 0, volumes[1:] / vol20avg, 1.0)

    # Align all to same length (drop first ~200 NaNs)
    n      = len(ret)
    offset = 200
    obs = np.column_stack([
        ret[offset:],
        np.nan_to_num(vol20[offset:], nan=0.01),
        trend[offset:],
        np.clip(vol_r[offset:], 0, 5),
        np.array([sg_vel(closes[max(0, i-20):i+1]) for i in range(offset, n)])
    ])
    return obs


# ── 4. DATA LOADER ────────────────────────────────────────────────────────────

def load_signal_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the pre-computed signal dataset produced by xgb_overlay_v2.py.
    Falls back to re-computing from ALL_SYMBOLS_OHLCV if signal CSV missing.
    """
    import os
    signal_csv = os.path.join(MODELS_DIR, 'signal_dataset.csv')
    if os.path.exists(signal_csv):
        import gc
        gc.collect()
        # rule_id is stored as strings ('rule1'…'rule11') — exclude from float32 dict
        _num = {c: 'float32' for c in FEATURE_COLS if c != 'rule_id'}
        _num.update({'hit_t1': 'float32', 'win': 'float32',
                     'win_rr': 'float32', 'rr_ratio': 'float32'})
        try:
            # memory_map avoids the C parser's large contiguous buffer allocation
            df = pd.read_csv(signal_csv, dtype=_num, memory_map=True)
        except (MemoryError, OSError, Exception):
            # Final fallback: no dtype hints, let pandas infer
            df = pd.read_csv(signal_csv, memory_map=True)
        # Encode rule_id: 'rule1' → 1, 'rule11' → 11, already int → keep
        if 'rule_id' in df.columns and df['rule_id'].dtype == object:
            df['rule_id'] = df['rule_id'].str.replace('rule', '', regex=False).astype(int)
        print(f"[data_utils] Loaded {len(df)} signals from cache.")
        return df
    # Cannot auto-compute here without running full backtest -- user should
    # run backtest_v3.py / xgb_overlay_v2.py first to generate the dataset.
    raise FileNotFoundError(
        f"Signal dataset not found at {signal_csv}.\n"
        "Run xgb_overlay_v2.py first with EXPORT_DATASET=True, or\n"
        "run scripts/ml/build_dataset.py."
    )
