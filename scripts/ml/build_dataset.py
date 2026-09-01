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
          win_rr  (1/0)  -- risk-adjusted win (return > 1.5x ATR%),
          rr_ratio (float) -- R-multiple (actual_return / TRAIL_STOP)
"""

import os, sys, warnings
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

RULE_MAP = {f'rule{i}': i for i in range(1, 12)}


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


def check_rules(cpr, prev_close, cur_close, ph, pl, vwap, cam):
    R = {}
    s3_in = cpr['lower'] <= cam['s3'] <= cpr['upper']
    r3_in = cpr['lower'] <= cam['r3'] <= cpr['upper']
    R['rule1']  = s3_in or r3_in
    R['rule2']  = cpr['width_pct'] < 0.5
    R['rule3']  = prev_close < cpr['upper'] and cur_close > cpr['upper']
    R['rule4']  = ph < cpr['lower'] or pl > cpr['upper']
    margin      = max(cpr['width'] * 0.5, cpr['pivot'] * 0.002)
    R['rule5']  = (cpr['lower'] - margin) <= vwap <= (cpr['upper'] + margin)
    safe        = cur_close if cur_close > 0 else 1
    nr3 = abs(cur_close - cam['r3']) / safe < 0.005
    ns3 = abs(cur_close - cam['s3']) / safe < 0.005
    R['rule6']  = cpr['width_pct'] > 0.7 and (nr3 or ns3)
    rs  = (prev_close > cpr['upper'] and cur_close > cpr['upper'] and
           (cur_close - cpr['upper']) / cpr['upper'] < 0.012)
    rr  = (prev_close < cpr['lower'] and cur_close < cpr['lower'] and
           (cpr['lower'] - cur_close) / cpr['lower'] < 0.012)
    R['rule7']  = rs or rr
    R['rule8']  = (cur_close > cpr['upper'] and ph > cpr['upper'])
    R['rule9']  = cpr['pivot'] > 0 and abs(cur_close - cpr['pivot']) / cpr['pivot'] > 0.02
    ht = cpr['upper'] > 0 and abs(ph - cpr['upper']) / cpr['upper'] < 0.005
    lt = cpr['lower'] > 0 and abs(pl - cpr['lower']) / cpr['lower'] < 0.005
    R['rule10'] = ht or lt
    R['rule11'] = ((cur_close > vwap and cur_close < cpr['upper']) or
                   (cur_close < vwap and cur_close > cpr['lower']))
    return R


def get_direction(rid, cpr, cam, cur_close, prev_close, ph, pl):
    safe = cur_close if cur_close > 0 else 1
    if rid == 'rule3':  return 1
    if rid == 'rule4':  return 1 if pl > cpr['upper'] else -1
    if rid == 'rule6':  return -1 if abs(cur_close - cam['r3']) / safe < 0.005 else 1
    if rid == 'rule7':  return 1 if prev_close > cpr['upper'] else -1
    if rid == 'rule8':  return 1 if cur_close > cpr['upper'] else -1
    if rid == 'rule9':  return -1 if cur_close > cpr['pivot'] else 1
    if rid == 'rule10':
        ht = cpr['upper'] > 0 and abs(ph - cpr['upper']) / cpr['upper'] < 0.005
        return -1 if ht else 1
    return 1 if cur_close >= cpr['pivot'] else -1


def build_signals_for_symbol(sym, df_sym, sector_closes=None,
                              delivery_pivot=None, pcr_pivot=None,
                              market_pcr=None,
                              vix_dict=None, sector_ticker='^NSEI'):
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

        rules_fired = check_rules(cpr, prev_close, cur_close, ph, pl, vwap_cur, cam)
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

        for rid in fired:
            direction  = get_direction(rid, cpr, cam, cur_close, prev_close, ph, pl)
            # Tier 1 interaction features that depend on direction
            rsi_dir  = rsi_val * direction
            hi52_dir = dist_hi52 * direction
            actual_ret = asymmetric_exit(direction, entry, fh, fl, fc) if entry > 0 else 0.0

            # --- Label definitions ---
            # win     : any positive move > 0.5% (legacy, kept for backward compat)
            # hit_t1  : full breakout to PROFIT_TARGET (cleanest signal)
            # win_rr  : return beats 1.5x ATR% (normalised across volatility regimes)
            # rr_ratio: R-multiple relative to initial stop (continuous target)
            hit_t1  = 1 if actual_ret >= PROFIT_TARGET * 0.90 else 0
            win_rr  = 1 if actual_ret / max(atr_pct, 0.001) > 0.8 else 0  # was 1.5 — too high vs T1 cap
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
                # --- labels ---
                'atr_pct':       round(atr_pct, 6),
                'actual_return': actual_ret,
                'win':           1 if actual_ret > 0.005 else 0,
                'hit_t1':        hit_t1,
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

    all_rows = []
    symbols  = df_all['Symbol'].unique()
    for i, sym in enumerate(symbols):
        if i % 100 == 0:
            print(f"  {i}/{len(symbols)} -- {sym}")
        df_sym = df_all[df_all['Symbol'] == sym][['Open','High','Low','Close','Volume']]
        rows   = build_signals_for_symbol(sym, df_sym,
                                          sector_closes=sector_closes,
                                          delivery_pivot=delivery_pivot,
                                          pcr_pivot=pcr_pivot,
                                          market_pcr=market_pcr,
                                          vix_dict=vix_dict)
        all_rows.extend(rows)

    df_signals = pd.DataFrame(all_rows)
    out = os.path.join(MODELS_DIR, 'signal_dataset.csv')
    df_signals.to_csv(out, index=False)
    print(f"\nSaved {len(df_signals)} signals -> {out}")
    print(f"  win     rate: {df_signals['win'].mean():.1%}   (actual_ret > 0.5%)")
    print(f"  hit_t1  rate: {df_signals['hit_t1'].mean():.1%}   (full T1 hit)")
    print(f"  win_rr  rate: {df_signals['win_rr'].mean():.1%}   (return > 1.5x ATR%)")
    print(f"  rr_ratio mean: {df_signals['rr_ratio'].mean():.3f}R")


if __name__ == '__main__':
    main()
