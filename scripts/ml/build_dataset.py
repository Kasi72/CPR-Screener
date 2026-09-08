"""
build_dataset.py -- Re-runs the backtest and exports signal_dataset.csv
used by all ML training phases.

Run ONCE before any training phase:
    python scripts/ml/build_dataset.py

Output: models/signal_dataset.csv  (one row per fired signal)
Columns: date, symbol, rule_id, direction, cpr_width_pct, vwap_dist,
          atr_pct_rank, vol_rank, n_rules_fired, sg_vel, ema200_dist,
          rsi14, mom5, dow, actual_return, atr_pct,
          win (1/0),
          hit_t1  (1/0)  -- T1 hit (return >= 90% of PROFIT_TARGET),
          hit_t3  (1/0)  -- T1 hit within 3 days (multi-day label),
          win_rr  (1/0)  -- risk-adjusted win (return > 0.8x ATR%),
          rr_ratio (float) -- R-multiple (actual_return / TRAIL_STOP),
          --- Sprint 1 CPR features ---
          cpr_overlap_pct    -- overlap between today/yesterday CPR bands [0,1]
          open_to_cpr_dist   -- direction-adj (entry - cpr_pivot) / ATR
          prev_cpr_respected -- 1 if price touched CPR but didn't break in last 3 days
          cpr_zone_vol_ratio -- proxy vol traded within CPR zone (prev bar overlap × vol_rank)
          hmm_regime         -- HMM market regime 0-3 (from hmm_posteriors.json, -1 if absent)
"""

import os, sys, warnings, json
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.ml.data_utils import (
    DATA_FILE, MODELS_DIR, ema, rsi_wilder, sg_vel, atr, atr_pct_rank,
    calc_cpr, calc_vwap, expiry_dist_days
)
from scripts.ml.sector_features import load_sector_closes, get_rs_features
from scripts.ml.download_bhavcopy import load_delivery_data
from scripts.ml.download_pcr import load_pcr_data, load_market_pcr

PROFIT_TARGET   = 0.025
TRAIL_STOP      = 0.008
MAX_HOLD        = 5
MIN_BARS        = 60
MIN_RULES_FIRED = 2    # confluence gate: skip single-rule signals (weaker, noisier)

RULE_MAP = {f'rule{i}': i for i in range(1, 17)}


def asymmetric_exit(direction, entry, fh, fl, fc):
    peak = 0.0
    HARD_STOP = -0.008   # -0.8% hard stop applies every bar (not just day 0)
    for d in range(len(fc)):
        best  = (fh[d] - entry) / entry if direction == 1 else (entry - fl[d]) / entry
        worst = (fl[d] - entry) / entry if direction == 1 else (entry - fh[d]) / entry
        # T1 hit
        if best >= PROFIT_TARGET:
            return PROFIT_TARGET * 0.97
        # Trailing stop (once we have a small profit cushion)
        peak = max(peak, best)
        if peak > 0.003 and (peak - best) >= TRAIL_STOP:
            return peak - TRAIL_STOP
        # Hard stop every bar (was incorrectly only on day 0 before)
        if worst < HARD_STOP:
            return worst
    return direction * (fc[-1] - entry) / entry if len(fc) > 0 else 0.0


def check_rules(cpr, prev_close, cur_close, ph, pl, vwap, cam,
                prev_vwap=None, hmm_regime=-1):
    R = {}
    s3_in = cpr['lower'] <= cam['s3'] <= cpr['upper']
    r3_in = cpr['lower'] <= cam['r3'] <= cpr['upper']
    R['rule1']  = s3_in or r3_in
    R['rule2']  = cpr['width_pct'] < 0.5
    R['rule3']  = prev_close < cpr['upper'] and cur_close > cpr['upper']
    R['rule4']  = ph < cpr['lower'] or pl > cpr['upper']
    # rule5: VWAP crossing CPR boundary (directional signal, not "VWAP inside CPR" state)
    if prev_vwap is not None:
        vwap_cross_up   = prev_vwap < cpr['lower'] and vwap >= cpr['lower']
        vwap_cross_down = prev_vwap > cpr['upper'] and vwap <= cpr['upper']
        R['rule5'] = vwap_cross_up or vwap_cross_down
    else:
        R['rule5'] = False
    # rule6: dropped — negative Sharpe (-1.61), confirmed losing strategy
    R['rule6']  = False
    rs  = (prev_close > cpr['upper'] and cur_close > cpr['upper'] and
           (cur_close - cpr['upper']) / cpr['upper'] < 0.012)
    rr  = (prev_close < cpr['lower'] and cur_close < cpr['lower'] and
           (cpr['lower'] - cur_close) / cpr['lower'] < 0.012)
    R['rule7']  = rs or rr
    R['rule8']  = (cur_close > cpr['upper'] and ph > cpr['upper'])
    # rule9: mean-reversion only valid in range-bound regimes (HMM 0 or 1)
    # trending regimes (2/3) destroy mean-reversion edge
    R['rule9']  = (cpr['pivot'] > 0
                   and abs(cur_close - cpr['pivot']) / cpr['pivot'] > 0.02
                   and hmm_regime in (0, 1))
    ht = cpr['upper'] > 0 and abs(ph - cpr['upper']) / cpr['upper'] < 0.005
    lt = cpr['lower'] > 0 and abs(pl - cpr['lower']) / cpr['lower'] < 0.005
    R['rule10'] = ht or lt
    R['rule11'] = ((cur_close > vwap and cur_close < cpr['upper']) or
                   (cur_close < vwap and cur_close > cpr['lower']))
    return R


def get_direction(rid, cpr, cam, cur_close, prev_close, ph, pl, prev_vwap=None, vwap=None):
    safe = cur_close if cur_close > 0 else 1
    if rid == 'rule3':  return 1
    if rid == 'rule4':  return 1 if pl > cpr['upper'] else -1
    if rid == 'rule5':
        # Cross up through BC = VWAP reclaim = long; cross down through TC = short
        if prev_vwap is not None and vwap is not None:
            return 1 if prev_vwap < cpr['lower'] else -1
        return 1
    if rid == 'rule6':  return 1  # dropped — will never fire
    if rid == 'rule7':  return 1 if prev_close > cpr['upper'] else -1
    if rid == 'rule8':  return 1 if cur_close > cpr['upper'] else -1
    if rid == 'rule9':  return -1 if cur_close > cpr['pivot'] else 1
    if rid == 'rule10':
        ht = cpr['upper'] > 0 and abs(ph - cpr['upper']) / cpr['upper'] < 0.005
        return -1 if ht else 1
    # rule12: Virgin CPR — approaching TC from below=short (resistance), approaching BC from above=long (support)
    if rid == 'rule12': return -1 if cur_close > cpr['pivot'] else 1
    # rule13: Squeeze Breakout — direction by price side of CPR
    if rid == 'rule13': return 1 if cur_close > cpr['upper'] else -1
    # rule14: Gap-Over-CPR — direction by price side of CPR
    if rid == 'rule14': return 1 if cur_close > cpr['upper'] else -1
    # rule15: Weekly CPR break — set externally via weekly_price_above_wtc flag (default long)
    if rid == 'rule15': return 1 if cur_close >= cpr['pivot'] else -1
    # rule16: Multi-factor confluence — direction by CPR side
    if rid == 'rule16': return 1 if cur_close >= cpr['pivot'] else -1
    return 1 if cur_close >= cpr['pivot'] else -1


def build_signals_for_symbol(sym, df_sym, sector_closes=None,
                              delivery_pivot=None, pcr_pivot=None,
                              market_pcr=None,
                              vix_dict=None, sector_ticker='^NSEI',
                              regime_map=None):
    df  = df_sym.sort_index().copy()
    if len(df) < MIN_BARS:
        return []

    # Liquidity filter: drop penny stocks (< ₹50) and illiquid stocks (< 50K vol/day)
    _quick_closes  = df['Close'].values
    _quick_volumes = df.get('Volume', pd.Series(np.ones(len(df)))).values
    if np.median(_quick_closes) < 50.0 or np.median(_quick_volumes) < 50_000.0:
        return []

    opens   = df['Open'].values
    highs   = df['High'].values
    lows    = df['Low'].values
    closes  = df['Close'].values
    volumes = df.get('Volume', pd.Series(np.ones(len(df)))).values
    dates   = df.index.tolist()

    # Build a Series for RS computation (date-indexed)
    sym_close_series = pd.Series(closes, index=pd.DatetimeIndex(dates))

    # Precompute CPR width per bar for 252-day percentile feature (once per symbol)
    _all_cpr_widths = np.array([
        calc_cpr(highs[j], lows[j], closes[j])['width_pct']
        for j in range(len(df) - 1)
    ], dtype=np.float32)

    # Precompute DatetimeIndex once — used by weekly CPR mask inside bar loop
    dates_dti = pd.DatetimeIndex(dates)

    rows = []
    for i in range(30, len(df) - MAX_HOLD - 1):
        H_prev, L_prev, C_prev = highs[i-1], lows[i-1], closes[i-1]
        cpr  = calc_cpr(H_prev, L_prev, C_prev)
        r    = H_prev - L_prev
        cam  = dict(r3=C_prev + r*1.1/4, s3=C_prev - r*1.1/4)

        cur_close  = closes[i]
        prev_close = closes[i-1]
        ph = highs[max(0, i-252):i].max()   # 52-week high (not all-time)
        pl = lows[max(0, i-252):i].min()    # 52-week low
        vwap_vals  = calc_vwap(opens[:i+1], highs[:i+1], lows[:i+1],
                                closes[:i+1], volumes[:i+1])
        vwap_cur   = vwap_vals[-1]
        vwap_prev  = vwap_vals[-2] if len(vwap_vals) >= 2 else vwap_cur

        # HMM regime lookup for rule9 gate
        if regime_map is not None:
            try:
                bar_dt     = pd.Timestamp(dates[i]).normalize()
                bar_regime = int(regime_map.asof(bar_dt))
            except Exception:
                bar_regime = -1
        else:
            bar_regime = -1

        rules_fired = check_rules(cpr, prev_close, cur_close, ph, pl, vwap_cur, cam,
                                  prev_vwap=vwap_prev, hmm_regime=bar_regime)
        fired = [k for k, v in rules_fired.items() if v]
        if not fired:
            continue

        # Confluence gate: require at least MIN_RULES_FIRED rules to confirm the setup
        if len(fired) < MIN_RULES_FIRED:
            continue

        # Technical indicators
        win_len = min(i + 1, 252)
        atrs_w  = [atr(highs[max(0,j-14):j], lows[max(0,j-14):j],
                       closes[max(0,j-14):j]) for j in range(max(14, i-win_len), i+1)]
        cur_atr   = atrs_w[-1] if atrs_w else 0.001
        atr_rank  = float(np.mean(np.array(atrs_w) <= cur_atr)) if atrs_w else 0.5

        vol20avg  = volumes[max(0, i-20):i].mean() if i > 0 else 1
        vol_rank  = float(volumes[i] / vol20avg) if vol20avg > 0 else 1.0
        rsi_val   = rsi_wilder(closes[max(0, i-29):i+1])
        sgv       = sg_vel(closes[max(0, i-20):i+1])
        ema200v   = ema(closes[max(0, i-200):i+1], 200)[-1] if i >= 200 else closes[i]
        ema200_dist = (cur_close - ema200v) / ema200v if ema200v > 0 else 0.0
        mom5      = (cur_close / closes[max(0, i-5)] - 1) if i >= 5 else 0.0
        vwap_dist = (cur_close - vwap_cur) / vwap_cur if vwap_cur > 0 else 0.0
        cpr_w     = cpr['width_pct']
        dow       = pd.Timestamp(dates[i]).dayofweek

        # --- Phase A: 52-week high/low distance + volume acceleration ---
        hi52      = highs[max(0, i-252):i].max() if i > 0 else cur_close
        lo52      = lows[max(0, i-252):i].min()  if i > 0 else cur_close
        dist_hi52 = (cur_close - hi52) / hi52 if hi52 > 0 else 0.0   # <=0 below 52w high
        dist_lo52 = (cur_close - lo52) / lo52 if lo52 > 0 else 0.0   # >=0 above 52w low
        vol5avg   = volumes[max(0, i-5):i].mean() if i > 0 else 1
        vol_accel = float(volumes[i] / vol5avg) if vol5avg > 0 else 1.0
        vol_accel = min(vol_accel, 10.0)

        # --- Phase B: Market / Sector Relative Strength ---
        if sector_closes:
            market_rs_5d, market_rs_20d, sector_rs_5d, sector_rs_20d = get_rs_features(
                sym_close_series, i, sector_closes, sector_ticker)
        else:
            market_rs_5d = market_rs_20d = sector_rs_5d = sector_rs_20d = 1.0

        # --- Phase C: Delivery % from NSE bhavcopy ---
        sig_date   = pd.Timestamp(dates[i]).normalize()
        deliv_pct  = 0.0
        if delivery_pivot is not None:
            try:
                if sym in delivery_pivot.columns and sig_date in delivery_pivot.index:
                    v = delivery_pivot.at[sig_date, sym]
                    deliv_pct = 0.0 if pd.isna(v) else float(v)
            except Exception:
                pass

        # --- Phase D: Options PCR (put-call ratio) ---
        pcr = 1.0   # neutral default
        if pcr_pivot is not None:
            try:
                if sym in pcr_pivot.columns and sig_date in pcr_pivot.index:
                    v = pcr_pivot.at[sig_date, sym]
                    if not pd.isna(v):
                        pcr = float(np.clip(v, 0.0, 10.0))
                    elif market_pcr is not None and sig_date in market_pcr.index:
                        # Fallback to market-wide Nifty PCR when stock PCR missing
                        mv = market_pcr.at[sig_date]
                        pcr = 1.0 if pd.isna(mv) else float(np.clip(mv, 0.0, 10.0))
            except Exception:
                pass
        elif market_pcr is not None:
            try:
                if sig_date in market_pcr.index:
                    mv = market_pcr.at[sig_date]
                    pcr = 1.0 if pd.isna(mv) else float(np.clip(mv, 0.0, 10.0))
            except Exception:
                pass

        # --- India VIX (market fear gauge) ---
        sig_date_str = str(dates[i])[:10]
        india_vix = float((vix_dict or {}).get(sig_date_str, 15.0))   # 15 = median neutral

        # --- Tier 2A: CPR quality features ---
        # CPR compress: today's CPR width / 5-day avg (squeeze = breakout setup)
        cpr_widths_5d = []
        for j in range(i - 5, i):
            if j >= 1:
                _cpj = calc_cpr(highs[j-1], lows[j-1], closes[j-1])
                cpr_widths_5d.append(_cpj['width_pct'])
        cpr_5d_avg   = np.mean(cpr_widths_5d) if cpr_widths_5d else cpr_w
        cpr_compress  = float(np.clip(cpr_w / cpr_5d_avg if cpr_5d_avg > 0 else 1.0, 0.0, 5.0))
        cpr_pos       = float(np.clip((cur_close - cpr['lower']) / cpr['width']
                                      if cpr['width'] > 0 else 0.5, 0.0, 1.0))
        dist_r1 = float(np.clip((cur_close - cpr['r1']) / cur_close
                                if cur_close > 0 else 0.0, -0.15, 0.15))
        dist_s1 = float(np.clip((cur_close - cpr['s1']) / cur_close
                                if cur_close > 0 else 0.0, -0.15, 0.15))

        # --- Tier 2B: multi-timeframe momentum ---
        mom3  = float((cur_close / closes[max(0, i-3)]  - 1) if i >= 3  else 0.0)
        mom10 = float((cur_close / closes[max(0, i-10)] - 1) if i >= 10 else 0.0)
        mom20 = float((cur_close / closes[max(0, i-20)] - 1) if i >= 20 else 0.0)

        # --- Tier 2C: RSI divergence + volume acceleration delta ---
        rsi_val_5ago = rsi_wilder(closes[max(0, i-34):max(1, i-4)]) if i >= 34 else 50.0
        price_ret_5  = (cur_close / closes[max(0, i-5)] - 1) if i >= 5 else 0.0
        rsi_chg_5    = rsi_val - rsi_val_5ago
        if abs(price_ret_5) > 0.005 and abs(rsi_chg_5) > 1.0:
            if (price_ret_5 > 0) and (rsi_chg_5 < 0):
                rsi_div = -1.0   # bearish divergence: price up, RSI down
            elif (price_ret_5 < 0) and (rsi_chg_5 > 0):
                rsi_div = 1.0    # bullish divergence: price down, RSI up
            else:
                rsi_div = 0.0
        else:
            rsi_div = 0.0

        vol5avg_prev = volumes[max(0, i-6):max(1, i-1)].mean() if i > 1 else 1
        vol_accel_prev = float(min(volumes[i-1] / vol5avg_prev if vol5avg_prev > 0 else 1.0, 10.0))
        vol_accel_delta = float(np.clip(vol_accel - vol_accel_prev, -5.0, 5.0))

        # --- Tier 2D: context features ---
        hi52_arr      = highs[max(0, i-252):i]
        if len(hi52_arr) > 0:
            last_hi_rel   = len(hi52_arr) - 1 - int(np.argmax(hi52_arr))
            days_since_52hi = float(min(last_hi_rel, 252))
        else:
            days_since_52hi = 252.0
        expiry_dist = float(expiry_dist_days(dates[i]))

        # --- Tier 1: conf_vol (no direction needed) ---
        conf_vol  = float(len(fired)) * vol_accel

        # Future returns for label
        fh = highs[i+1:i+MAX_HOLD+1]
        fl = lows[i+1:i+MAX_HOLD+1]
        fc = closes[i+1:i+MAX_HOLD+1]
        entry = opens[i+1] if i+1 < len(opens) else closes[i]

        # ATR as % of current close -- used for risk-adjusted label
        atr_pct = float(cur_atr / cur_close) if cur_close > 0 else 0.001

        # --- Sprint 1: New CPR features (per-bar, direction-independent) ---

        # 1. CPR overlap with yesterday's CPR bands
        if i >= 2:
            cpr_prev2    = calc_cpr(highs[i-2], lows[i-2], closes[i-2])
            overlap_abs  = max(0.0, min(cpr['upper'], cpr_prev2['upper'])
                                   - max(cpr['lower'], cpr_prev2['lower']))
            cpr_overlap_pct = float(np.clip(
                overlap_abs / max(cpr['width'], 0.0001), 0.0, 1.0))
        else:
            cpr_overlap_pct = 0.5

        # 2. Volume-in-CPR-zone proxy: prev bar's H-L overlap with CPR × vol_rank
        prev_range       = max(highs[i-1] - lows[i-1], 0.001)
        cpr_bar_overlap  = max(0.0, min(highs[i-1], cpr['upper'])
                                    - max(lows[i-1], cpr['lower']))
        cpr_zone_vol_ratio = float(np.clip(
            (cpr_bar_overlap / prev_range) * vol_rank, 0.0, 5.0))

        # 3. Previous CPR respected: price touched CPR in last 3 days but didn't close through
        prev_cpr_respected = 0.0
        for _j in range(max(1, i - 3), i):
            _cpj     = calc_cpr(highs[_j-1], lows[_j-1], closes[_j-1])
            _touched = lows[_j] <= _cpj['upper'] and highs[_j] >= _cpj['lower']
            _broke   = (_cpj['width'] > 0 and
                        abs(closes[_j] - _cpj['pivot']) > _cpj['width'] * 1.5)
            if _touched and not _broke:
                prev_cpr_respected = 1.0
                break

        # --- Sprint 2: High-lift CPR features (shared 20-day lookback) ---

        # One pass: CPR data for last 20 bars + today
        _lb_start = max(1, i - 19)
        _cpr_lb   = []
        for _j in range(_lb_start, i + 1):
            _c = calc_cpr(highs[_j-1], lows[_j-1], closes[_j-1])
            _cpr_lb.append({'width_pct': _c['width_pct'], 'upper': _c['upper'],
                            'lower': _c['lower'],
                            'mid': (_c['upper'] + _c['lower']) / 2.0})
        # _cpr_lb[-1] = today's CPR (matches `cpr`)

        # 5. open_inside_cpr: today's open between BC and TC
        open_inside_cpr = 1.0 if cpr['lower'] <= opens[i] <= cpr['upper'] else 0.0

        # 6. cpr_virgin: no bar in last 20 days touched today's CPR zone
        cpr_virgin = 1.0
        for _j in range(max(0, i - 20), i):
            if lows[_j] <= cpr['upper'] and highs[_j] >= cpr['lower']:
                cpr_virgin = 0.0
                break

        # 7. consecutive_narrow_cprs: streak of CPR widths below 20-day median
        _widths_hist = [x['width_pct'] for x in _cpr_lb[:-1]]
        _median_w    = float(np.median(_widths_hist)) if _widths_hist else cpr_w
        _consec      = 0
        for _x in reversed(_widths_hist):
            if _x < _median_w:
                _consec += 1
            else:
                break
        consecutive_narrow_cprs = float(min(_consec, 10))

        # 8. cpr_midpoint_trend: 5-day slope of midpoints / ATR
        _mids = [x['mid'] for x in _cpr_lb[-6:-1]]
        if len(_mids) >= 2:
            _x_arr = np.arange(len(_mids), dtype=np.float32)
            _slope = float(np.polyfit(_x_arr, _mids, 1)[0])
            cpr_midpoint_trend = float(np.clip(_slope / max(cur_atr, 0.001), -5.0, 5.0))
        else:
            cpr_midpoint_trend = 0.0

        # 9. cpr_expansion_factor: today's CPR width / yesterday's
        if len(_cpr_lb) >= 2:
            cpr_expansion_factor = float(np.clip(
                cpr_w / max(_cpr_lb[-2]['width_pct'], 0.001), 0.1, 10.0))
        else:
            cpr_expansion_factor = 1.0

        # --- Sprint 3: Directional gap + bar quality + volatility/volume structure ---

        # S3-1. gap_pct: today's open vs prev close (gap direction = continuation signal)
        gap_pct = float(np.clip((opens[i] - closes[i-1]) / closes[i-1], -0.10, 0.10))

        # S3-2. cpr_test_count_5d: how many of last 5 bars touched today's CPR zone (magnetism)
        cpr_test_count_5d = 0
        for _j in range(max(1, i-5), i):
            if lows[_j] <= cpr['upper'] and highs[_j] >= cpr['lower']:
                cpr_test_count_5d += 1

        # S3-3. prev_bar_close_pos: yesterday's close as fraction of its H-L range [0=low, 1=high]
        _prev_hl = highs[i-1] - lows[i-1]
        prev_bar_close_pos = float(np.clip(
            (closes[i-1] - lows[i-1]) / max(_prev_hl, 0.001), 0.0, 1.0))

        # S3-4. atr_expansion: today's ATR / 5-day avg ATR (>1=expanding, <1=contracting)
        _atrs_5d = [atr(highs[max(0, _j-14):_j], lows[max(0, _j-14):_j],
                        closes[max(0, _j-14):_j]) for _j in range(max(14, i-5), i)]
        _avg_atr_5d = float(np.mean(_atrs_5d)) if _atrs_5d else cur_atr
        atr_expansion = float(np.clip(cur_atr / max(_avg_atr_5d, 0.001), 0.2, 5.0))

        # S3-5. vol_trend_slope: 5-day volume slope / avg (positive = volume building)
        _vol5 = volumes[max(0, i-5):i].astype(float)
        if len(_vol5) >= 3:
            _vmean = _vol5.mean()
            _vx    = np.arange(len(_vol5), dtype=float)
            vol_trend_slope = float(np.clip(
                np.polyfit(_vx, _vol5, 1)[0] / max(_vmean, 1), -1.0, 1.0))
        else:
            vol_trend_slope = 0.0

        # --- Sprint 2 (Part B): structural + context CPR features ---

        # 10. cpr_above_prev_cpr: today's BC > yesterday's TC (bullish CPR structure gap)
        if len(_cpr_lb) >= 2:
            cpr_above_prev_cpr = 1.0 if cpr['lower'] > _cpr_lb[-2]['upper'] else 0.0
        else:
            cpr_above_prev_cpr = 0.0

        # 11. prev_close_inside_cpr: yesterday's close inside today's CPR (indecision → explosion)
        prev_close_inside_cpr = 1.0 if cpr['lower'] <= closes[i-1] <= cpr['upper'] else 0.0

        # 12. atr_to_cpr_ratio: ATR% / CPR_width% (elastic-band breakout ratio)
        atr_to_cpr_ratio = float(np.clip(atr_pct / max(cpr_w, 0.001), 0.1, 20.0))

        # 13. cpr_width_percentile_252d: 1 = narrowest (max compression), 0 = widest
        _w252 = _all_cpr_widths[max(0, i - 252): i]
        cpr_width_percentile_252d = float(1.0 - np.mean(_w252 <= cpr_w)) if len(_w252) > 5 else 0.5

        # 14. prev_day_ochoa_type: Frank Ochoa day classification for yesterday
        #     0=Trend (close beyond R1/S1), 1=Normal, 2=Neutral (inside CPR), 3=Outside (wide range)
        if i >= 2:
            _cpr_pd   = calc_cpr(highs[i-2], lows[i-2], closes[i-2])
            _prev_atr = atr(highs[max(0, i-15):i-1], lows[max(0, i-15):i-1], closes[max(0, i-15):i-1])
            _prev_rng = highs[i-1] - lows[i-1]
            _pc       = closes[i-1]
            if _pc > _cpr_pd['r1'] or _pc < _cpr_pd['s1']:
                prev_day_ochoa_type = 0
            elif _cpr_pd['lower'] <= _pc <= _cpr_pd['upper']:
                prev_day_ochoa_type = 2
            elif _prev_rng > max(_prev_atr, 0.001) * 1.5:
                prev_day_ochoa_type = 3
            else:
                prev_day_ochoa_type = 1
        else:
            prev_day_ochoa_type = 1

        # ── Weekly CPR computation ────────────────────────────────────────────────
        cur_dt     = pd.Timestamp(dates[i])
        week_start = cur_dt - pd.Timedelta(days=cur_dt.dayofweek)  # Monday of current week
        prior_week_mask = (
            (dates_dti >= week_start - pd.Timedelta(days=7))
            & (dates_dti < week_start)
        )
        prior_week_idx = np.where(prior_week_mask)[0]
        weekly_cpr_first_break = False
        weekly_price_above_wtc = cur_close >= cpr['pivot']
        if len(prior_week_idx) >= 3:
            w_H = highs[prior_week_idx].max()
            w_L = lows[prior_week_idx].min()
            w_C = closes[prior_week_idx[-1]]
            w_P  = (w_H + w_L + w_C) / 3.0
            w_TC = (w_H + w_L) / 2.0
            w_BC = 2.0 * w_P - w_TC
            weekly_price_above_wtc = cur_close > w_TC
            week_so_far_mask = (
                (dates_dti >= week_start)
                & (dates_dti <= cur_dt)
            )
            wsf_idx = np.where(week_so_far_mask)[0]
            if len(wsf_idx) > 1:
                prev_above_wtc = sum(closes[j] > w_TC for j in wsf_idx[:-1])
                prev_below_wbc = sum(closes[j] < w_BC for j in wsf_idx[:-1])
                weekly_cpr_first_break = (
                    (cur_close > w_TC and prev_above_wtc == 0) or
                    (cur_close < w_BC and prev_below_wbc == 0)
                )

        # ── New rules 12-16 (extra fired list, merged into fired before direction loop) ──
        _extra_fired = []

        # rule12: Virgin CPR Precision Test
        # First touch of 20-day-untouched CPR — institutional S/R with volume urgency
        if cpr_virgin == 1.0:
            near_tc = 0 < (cpr['upper'] - cur_close) / max(cpr['upper'], 1) < 0.004
            near_bc = 0 < (cur_close - cpr['lower']) / max(cpr['lower'], 1) < 0.004
            if (near_tc or near_bc) and vol_rank >= 0.85 and open_inside_cpr == 0.0 and prev_cpr_respected == 1.0:
                _extra_fired.append('rule12')

        # rule13: CPR Squeeze Breakout — 3+ narrow days then decisive expansion + vol surge
        is_squeeze_release = (
            consecutive_narrow_cprs >= 3
            and cpr_expansion_factor >= 1.4
            and cpr_width_percentile_252d >= 0.50
        )
        broke_upper = cur_close > cpr['upper'] and gap_pct > -0.005
        broke_lower = cur_close < cpr['lower'] and gap_pct < 0.005
        if is_squeeze_release and (broke_upper or broke_lower) and vol_rank >= 0.9 and vol_trend_slope > 0.0 and atr_expansion >= 1.15:
            _extra_fired.append('rule13')

        # rule14: Gap-Over-CPR Continuation — clean gap held all day + Ochoa trend yesterday
        gap_above = opens[i] > cpr['upper'] * 1.002
        gap_below = opens[i] < cpr['lower'] * 0.998
        held_above = gap_above and cur_close > cpr['upper']
        held_below = gap_below and cur_close < cpr['lower']
        dir_bull = gap_above and prev_close > cpr['upper']
        dir_bear = gap_below and prev_close < cpr['lower']
        if (held_above or held_below) and prev_day_ochoa_type == 0 and 0.80 <= vol_rank <= 3.5 and (dir_bull or dir_bear):
            _extra_fired.append('rule14')

        # rule15: Weekly CPR Breakout — first break of weekly TC/BC this week + Mon-Wed only
        if weekly_cpr_first_break and vol_rank >= 0.85 and market_rs_5d > 1.0 and cur_dt.dayofweek <= 2:
            _extra_fired.append('rule15')

        # rule16: Multi-Factor Confluence — all 8 factors aligned (bull or bear)
        bull_conf = (
            cur_close > cpr['upper'] and pcr < 0.85 and india_vix < 16.0
            and bar_regime in (1, 2) and market_rs_5d > 1.015
            and deliv_pct >= 38.0 and mom5 > 0.008 and ema200_dist > 0.0
        )
        bear_conf = (
            cur_close < cpr['lower'] and pcr > 1.15 and india_vix > 17.0
            and bar_regime in (2, 3) and market_rs_5d < 0.985
            and deliv_pct < 25.0 and mom5 < -0.008 and ema200_dist < -0.005
        )
        if bull_conf or bear_conf:
            _extra_fired.append('rule16')

        fired = fired + _extra_fired

        for rid in fired:
            direction  = get_direction(rid, cpr, cam, cur_close, prev_close, ph, pl,
                                       prev_vwap=vwap_prev, vwap=vwap_cur)
            # Tier 1 interaction features that depend on direction
            rsi_dir  = rsi_val * direction
            hi52_dir = dist_hi52 * direction
            actual_ret = asymmetric_exit(direction, entry, fh, fl, fc) if entry > 0 else 0.0

            # 4. Open-to-CPR distance (direction-adjusted: + = opened on favorable side)
            open_to_cpr_dist = float(np.clip(
                (entry - cpr['pivot']) / max(cur_atr, 0.001) * direction,
                -5.0, 5.0))

            # --- Label definitions ---
            # win     : any positive move > 0.5% (legacy, kept for backward compat)
            # hit_t1  : full breakout to PROFIT_TARGET (cleanest signal)
            # hit_t3  : T1 hit within 3 days (multi-day, less noise)
            # win_rr  : return beats 0.8x ATR% (normalised across volatility regimes)
            # rr_ratio: R-multiple relative to initial stop (continuous target)
            hit_t1  = 1 if actual_ret >= PROFIT_TARGET * 0.90 else 0
            hit_t3  = 1 if (len(fc) >= 3 and
                            asymmetric_exit(direction, entry, fh[:3], fl[:3], fc[:3])
                            >= PROFIT_TARGET * 0.90) else 0
            win_rr  = 1 if actual_ret / max(atr_pct, 0.001) > 0.8 else 0
            rr_ratio = round(actual_ret / TRAIL_STOP, 4)

            rows.append({
                # --- identifiers ---
                'date':          str(dates[i])[:10],
                'symbol':        sym,
                'rule_id':       rid,
                'direction':     direction,
                # --- original features ---
                'cpr_width_pct': cpr_w,
                'vwap_dist':     vwap_dist,
                'atr_pct_rank':  atr_rank,
                'vol_rank':      min(vol_rank, 5.0),
                'n_rules_fired': len(fired),
                'sg_vel':        sgv,
                'ema200_dist':   ema200_dist,
                'rsi14':         rsi_val,
                'mom5':          mom5,
                'dow':           dow,
                # --- Phase A: 52w + vol accel ---
                'dist_hi52':     round(dist_hi52, 5),
                'dist_lo52':     round(dist_lo52, 5),
                'vol_accel':     round(vol_accel, 4),
                # --- Phase B: relative strength ---
                'market_rs_5d':  round(market_rs_5d,  4),
                'market_rs_20d': round(market_rs_20d, 4),
                'sector_rs_5d':  round(sector_rs_5d,  4),
                'sector_rs_20d': round(sector_rs_20d, 4),
                # --- Phase C: delivery ---
                'deliv_pct':     round(deliv_pct, 2),
                # --- Phase D: options PCR ---
                'pcr':           round(pcr, 4),
                # --- Tier 1: VIX + interaction features ---
                'india_vix':     round(india_vix, 2),
                'conf_vol':      round(min(conf_vol, 50.0), 4),
                'rsi_dir':       round(rsi_dir, 2),
                'hi52_dir':      round(hi52_dir, 5),
                # --- Tier 2A: CPR quality ---
                'cpr_compress':  round(cpr_compress, 4),
                'cpr_pos':       round(cpr_pos, 4),
                'dist_r1':       round(dist_r1, 5),
                'dist_s1':       round(dist_s1, 5),
                # --- Tier 2B: multi-timeframe momentum ---
                'mom3':          round(mom3, 5),
                'mom10':         round(mom10, 5),
                'mom20':         round(mom20, 5),
                # --- Tier 2C: divergence + volume curvature ---
                'rsi_div':       rsi_div,
                'vol_accel_delta': round(vol_accel_delta, 4),
                # --- Tier 2D: context ---
                'days_since_52hi': days_since_52hi,
                'expiry_dist':   expiry_dist,
                # --- Sprint 1: new CPR alpha features ---
                'cpr_overlap_pct':    round(cpr_overlap_pct, 4),
                'open_to_cpr_dist':   round(open_to_cpr_dist, 4),
                'prev_cpr_respected': int(prev_cpr_respected),
                'cpr_zone_vol_ratio': round(cpr_zone_vol_ratio, 4),
                # --- Sprint 2A: compression/structure CPR features ---
                'open_inside_cpr':             int(open_inside_cpr),
                'cpr_virgin':                  int(cpr_virgin),
                'consecutive_narrow_cprs':     consecutive_narrow_cprs,
                'cpr_midpoint_trend':          round(cpr_midpoint_trend, 4),
                'cpr_expansion_factor':        round(cpr_expansion_factor, 4),
                # --- Sprint 2B: structural + context CPR features ---
                'cpr_above_prev_cpr':          int(cpr_above_prev_cpr),
                'prev_close_inside_cpr':       int(prev_close_inside_cpr),
                'atr_to_cpr_ratio':            round(atr_to_cpr_ratio, 4),
                'cpr_width_percentile_252d':   round(cpr_width_percentile_252d, 4),
                'prev_day_ochoa_type':         prev_day_ochoa_type,
                # --- Sprint 3: gap + bar quality + volatility + volume structure ---
                'gap_pct':                     round(gap_pct, 5),
                'cpr_test_count_5d':           cpr_test_count_5d,
                'prev_bar_close_pos':          round(prev_bar_close_pos, 4),
                'atr_expansion':               round(atr_expansion, 4),
                'vol_trend_slope':             round(vol_trend_slope, 4),
                # --- Sprint 4: weekly CPR features (for rule15) ---
                'weekly_cpr_first_break':      int(weekly_cpr_first_break),
                'weekly_price_above_wtc':      int(weekly_price_above_wtc),
                # --- labels ---
                'atr_pct':       round(atr_pct, 6),
                'actual_return': actual_ret,
                'win':           1 if actual_ret > 0.005 else 0,
                'hit_t1':        hit_t1,
                'hit_t3':        hit_t3,
                'win_rr':        win_rr,
                'rr_ratio':      rr_ratio,
            })
    return rows


def main():
    print(f"Loading OHLCV data from {DATA_FILE}...")
    df_all = pd.read_csv(DATA_FILE, low_memory=False)
    df_all.columns = [c.strip().title() for c in df_all.columns]
    df_all['Date'] = pd.to_datetime(df_all['Date'], dayfirst=True)
    # downcast numerics to float32 to halve RAM footprint before sort/index
    float_cols = df_all.select_dtypes(include='float64').columns
    df_all[float_cols] = df_all[float_cols].astype('float32')
    # inplace to avoid two full-dataframe copies (OOM on large CSVs)
    df_all.sort_values('Date', inplace=True)
    df_all.set_index('Date', inplace=True)
    print(f"Loaded {len(df_all)} rows, {df_all['Symbol'].nunique()} symbols")

    # --- Phase B: load sector closes (optional) ---
    sector_closes = load_sector_closes()
    if sector_closes:
        print(f"Sector data: {len(sector_closes)} indices loaded "
              f"(run sector_features.py to update)")
    else:
        print("Sector data not found -- market_rs / sector_rs = 1.0 (neutral).")
        print("  Run: python scripts/ml/sector_features.py  to download")

    # --- Phase C: load delivery data (optional) ---
    delivery_pivot = load_delivery_data()
    if delivery_pivot is None:
        print("Delivery data not found -- deliv_pct = 0.")
        print("  Run: python scripts/ml/download_bhavcopy.py  to download")

    # --- Phase D: load options PCR data (optional) ---
    pcr_pivot = load_pcr_data()
    market_pcr = load_market_pcr()
    if pcr_pivot is None and market_pcr is None:
        print("PCR data not found -- pcr = 1.0 (neutral).")
        print("  Run: python scripts/ml/download_pcr.py  to download")
    elif market_pcr is not None:
        print(f"Market PCR loaded: {len(market_pcr)} dates (Nifty OPTIDX fallback active)")

    # --- Tier 1: India VIX (download via yfinance) ---
    print("Downloading India VIX (^INDIAVIX)...")
    vix_dict = {}
    try:
        vix_df = yf.download('^INDIAVIX', start='2018-01-01', progress=False, auto_adjust=True)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)
        vix_series = vix_df['Close'].dropna()
        vix_dict = {str(d.date()): float(v) for d, v in vix_series.items()}
        print(f"  India VIX: {len(vix_dict)} days loaded "
              f"({min(vix_dict.keys())} to {max(vix_dict.keys())})")
    except Exception as e:
        print(f"  India VIX download failed ({e}) — using neutral 15.0 for all signals")

    # --- Sprint 1: load HMM posteriors once for regime assignment ---
    posteriors_path = os.path.join(MODELS_DIR, 'hmm_posteriors.json')
    regime_map = None
    stability_map = None  # max(posterior) — HMM certainty score [0.25, 1.0]
    risk_map = None       # normalized entropy — [0=certain, 1=max uncertainty]
    if os.path.exists(posteriors_path):
        with open(posteriors_path) as f:
            posteriors = json.load(f)

        _dates = list(posteriors.keys())
        _probs = [np.array(v, dtype=np.float32) for v in posteriors.values()]
        _n_states = _probs[0].shape[0] if _probs else 4
        _log_n = np.log(_n_states)

        _regime_series = pd.Series(
            {d: int(np.argmax(v)) for d, v in zip(_dates, _probs)},
            dtype='int8',
        )
        _stability_series = pd.Series(
            {d: float(v.max()) for d, v in zip(_dates, _probs)},
            dtype='float32',
        )
        # normalized entropy: 0 = fully certain, 1 = uniform (max uncertainty)
        _risk_series = pd.Series(
            {d: float(-np.sum(v * np.log(np.clip(v, 1e-9, 1))) / _log_n)
             for d, v in zip(_dates, _probs)},
            dtype='float32',
        )

        for s in (_regime_series, _stability_series, _risk_series):
            s.index = pd.to_datetime(s.index)
            s.sort_index(inplace=True)

        regime_map    = _regime_series
        stability_map = _stability_series
        risk_map      = _risk_series
        print(f"  HMM posteriors loaded: {len(regime_map)} dates  "
              f"(stability mean={_stability_series.mean():.3f}  "
              f"risk mean={_risk_series.mean():.3f})")
    else:
        print("  hmm_posteriors.json not found — hmm_regime = -1 (neutral).")

    # Stream rows directly to CSV to avoid accumulating 1.4M dicts in RAM
    out  = os.path.join(MODELS_DIR, 'signal_dataset.csv')
    symbols      = df_all['Symbol'].unique()
    total_rows   = 0
    header_done  = False

    for i, sym in enumerate(symbols):
        if i % 100 == 0:
            print(f"  {i}/{len(symbols)} -- {sym}")
        df_sym = df_all[df_all['Symbol'] == sym][['Open','High','Low','Close','Volume']]
        rows   = build_signals_for_symbol(sym, df_sym,
                                          sector_closes=sector_closes,
                                          delivery_pivot=delivery_pivot,
                                          pcr_pivot=pcr_pivot,
                                          market_pcr=market_pcr,
                                          vix_dict=vix_dict,
                                          regime_map=regime_map)
        if not rows:
            continue

        chunk = pd.DataFrame(rows)
        if regime_map is not None:
            dates_dt = pd.to_datetime(chunk['date'])
            _tol = pd.Timedelta('5D')
            # reindex to signal dates; ffill fills yfinance gaps (max 5 days)
            chunk['hmm_regime'] = (
                regime_map.reindex(dates_dt, method='ffill', tolerance=_tol)
                .fillna(-1).astype(int).values
            )
            chunk['regime_stability'] = (
                stability_map.reindex(dates_dt, method='ffill', tolerance=_tol)
                .fillna(0.25).astype(np.float32).values  # 0.25 = uniform (4 states)
            )
            chunk['transition_risk'] = (
                risk_map.reindex(dates_dt, method='ffill', tolerance=_tol)
                .fillna(1.0).astype(np.float32).values   # 1.0 = max uncertainty
            )
        else:
            chunk['hmm_regime']       = -1
            chunk['regime_stability'] = 0.25   # unknown → uniform prior
            chunk['transition_risk']  = 1.0    # unknown → max uncertainty

        chunk.to_csv(out, mode='w' if not header_done else 'a',
                     header=not header_done, index=False)
        header_done = True
        total_rows += len(chunk)

    # Read back for summary stats and Phase 2b/2c score injection (Sprint 4)
    print("\nReading CSV back for meta-feature injection ...")
    df_signals = pd.read_csv(out, low_memory=False)
    print(f"\nSaved {total_rows} signals -> {out}")
    print(f"  HMM regime dist: {df_signals['hmm_regime'].value_counts().to_dict()}")
    print(f"  win     rate: {df_signals['win'].mean():.1%}   (actual_ret > 0.5%)")
    print(f"  hit_t1  rate: {df_signals['hit_t1'].mean():.1%}   (full T1 hit)")
    print(f"  hit_t3  rate: {df_signals['hit_t3'].mean():.1%}   (T1 within 3 days)")
    print(f"  win_rr  rate: {df_signals['win_rr'].mean():.1%}   (return > 0.8x ATR%)")
    print(f"  rr_ratio mean: {df_signals['rr_ratio'].mean():.3f}R")

    # --- Sprint 4: inject Phase 2b + 2c scores as meta-features ---------------
    # These become base learner inputs for Phase 3 stacking meta-learner.
    # Both models are optional — skipped gracefully if not yet trained.

    P2B_FEATURE_COLS = [
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
        'regime_stability', 'transition_risk',
        # Sprint 4: weekly CPR (added to match phase2b kernel FEATURE_COLS = 40)
        'weekly_cpr_first_break', 'weekly_price_above_wtc',
    ]  # 40 — matches Phase 2b kernel FEATURE_COLS

    P2C_BASE_COLS = [
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
        # Sprint 2A: compression/structure
        'open_inside_cpr', 'cpr_virgin', 'consecutive_narrow_cprs',
        'cpr_midpoint_trend', 'cpr_expansion_factor',
        # Sprint 2B: structural + context
        'cpr_above_prev_cpr', 'prev_close_inside_cpr', 'atr_to_cpr_ratio',
        'cpr_width_percentile_252d', 'prev_day_ochoa_type',
        # Sprint 3: gap + bar quality + volatility + volume structure
        'gap_pct', 'cpr_test_count_5d', 'prev_bar_close_pos',
        'atr_expansion', 'vol_trend_slope',
        # Sprint 4: weekly CPR
        'weekly_cpr_first_break', 'weekly_price_above_wtc',
    ]  # 58 — matches Phase 2c kernel BASE_FEATURES

    injected = False
    try:
        import lightgbm as _lgb

        # Ensure string columns are numeric for LightGBM
        for col in ['rule_id', 'direction']:
            if col in df_signals.columns and df_signals[col].dtype == object:
                extracted = df_signals[col].astype(str).str.extract(r'(\d+)')[0]
                if extracted.notna().mean() > 0.5:
                    df_signals[col] = pd.to_numeric(extracted, errors='coerce').fillna(0)
                else:
                    df_signals[col] = df_signals[col].astype('category').cat.codes

        # Phase 2b score
        p2b_path = os.path.join(MODELS_DIR, 'lgbm_scorer.txt')
        if os.path.exists(p2b_path):
            p2b_cols_present = [c for c in P2B_FEATURE_COLS if c in df_signals.columns]
            missing_p2b      = [c for c in P2B_FEATURE_COLS if c not in df_signals.columns]
            for c in missing_p2b:
                df_signals[c] = 0.0
            p2b_model = _lgb.Booster(model_file=p2b_path)
            X_p2b = df_signals[P2B_FEATURE_COLS].fillna(0).values.astype(np.float32)
            df_signals['lgbm2b_score'] = np.clip(p2b_model.predict(X_p2b), 0.0, 1.0)
            print(f"  lgbm2b_score injected (mean={df_signals['lgbm2b_score'].mean():.4f})")
            injected = True
        else:
            df_signals['lgbm2b_score'] = 0.5
            print("  lgbm_scorer.txt not found — lgbm2b_score = 0.5 (neutral placeholder)")

        # Phase 2c interaction features (needed for 2c model input)
        df_signals['cpr_vol_interaction']    = (1.0 - df_signals['cpr_compress'].clip(0, 1)) * df_signals['vol_rank']
        df_signals['regime_momentum']        = df_signals['hmm_regime'].clip(0, 3) * df_signals['mom5']
        df_signals['cpr_rsi_squeeze']        = (1.0 - df_signals['cpr_width_pct'].clip(0, 1)) * df_signals['rsi14'] / 100.0
        df_signals['overlap_vol_signal']     = df_signals['cpr_overlap_pct'] * df_signals['cpr_zone_vol_ratio']
        df_signals['rs_direction_alignment'] = (df_signals['market_rs_5d'] + df_signals['sector_rs_5d']) * df_signals['direction']
        df_signals['virgin_momentum']        = df_signals.get('cpr_virgin', 0.0) * df_signals['mom5']
        df_signals['narrow_breakout_vol']    = df_signals.get('consecutive_narrow_cprs', 0.0) * df_signals['vol_rank']

        P2C_ALL_COLS = P2C_BASE_COLS + [
            'cpr_vol_interaction', 'regime_momentum', 'cpr_rsi_squeeze',
            'overlap_vol_signal', 'rs_direction_alignment',
            'virgin_momentum', 'narrow_breakout_vol',
        ]  # 65 (58 base + 7 interactions)

        # Phase 2c global score
        p2c_path = os.path.join(MODELS_DIR, 'lgbm2c_global.txt')
        if os.path.exists(p2c_path):
            for c in P2C_ALL_COLS:
                if c not in df_signals.columns:
                    df_signals[c] = 0.0
            p2c_model = _lgb.Booster(model_file=p2c_path)
            X_p2c = df_signals[P2C_ALL_COLS].fillna(0).values.astype(np.float32)
            df_signals['lgbm2c_score'] = np.clip(p2c_model.predict(X_p2c), 0.0, 1.0)
            print(f"  lgbm2c_score injected (mean={df_signals['lgbm2c_score'].mean():.4f})")
            injected = True
        else:
            df_signals['lgbm2c_score'] = 0.5
            print("  lgbm2c_global.txt not found — lgbm2c_score = 0.5 (neutral placeholder)")

        # Drop temporary interaction cols (they'll be recomputed by Phase 2c kernel)
        df_signals.drop(columns=['cpr_vol_interaction', 'regime_momentum', 'cpr_rsi_squeeze',
                                  'overlap_vol_signal', 'rs_direction_alignment',
                                  'virgin_momentum', 'narrow_breakout_vol'],
                        inplace=True, errors='ignore')

        # Overwrite CSV with meta-feature columns added
        df_signals.to_csv(out, index=False)
        if injected:
            print(f"  CSV updated with lgbm2b_score + lgbm2c_score → {out}")

    except Exception as e:
        print(f"  Sprint 4 meta-feature injection skipped: {e}")


if __name__ == '__main__':
    main()
