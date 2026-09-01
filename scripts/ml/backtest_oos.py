"""
backtest_oos.py - Out-of-Sample + Walk-Forward Backtest

Two tests:
  1. Static OOS  : trained lgbm_model.txt on last 20% of data (never seen during training)
  2. Walk-Forward: 5-fold expanding window, retrain LightGBM each fold, test on next fold

Metrics per fold + aggregate:
  - AUC, Win Rate, Profit Factor, Sharpe, Max Drawdown, Calmar, Total Signals

Usage:
    python scripts/ml/backtest_oos.py
    python scripts/ml/backtest_oos.py --folds 6 --min-prob 0.5

Output:
    models/backtest_oos_results.json
    models/backtest_oos_report.html
"""
# [STANDALONE] One-off analysis script — not part of the ML training pipeline.


import os, sys, json, argparse, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.ml.data_utils import MODELS_DIR, FEATURE_COLS, WIN_COL, load_signal_dataset

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# ── Config ────────────────────────────────────────────────────────────────────

LGBM_PARAMS = {
    'objective':        'binary',
    'metric':           ['binary_logloss', 'auc'],
    'num_leaves':       63,
    'learning_rate':    0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq':     5,
    'min_child_samples': 20,
    'verbose':          -1,
    'seed':             42,
}

PROFIT_TARGET = 0.025   # 2.5% T1
TRAIL_STOP    = 0.008   # 0.8% trailing stop


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_metrics(returns, labels, probs, min_prob=0.0):
    """
    Compute trading metrics for signals where model prob >= min_prob.
    labels  : array of actual WIN_COL values
    returns : array of actual_return
    probs   : model predicted probability
    """
    mask = probs >= min_prob
    r   = returns[mask]
    lab = labels[mask]
    n   = len(r)

    if n == 0:
        return {'n': 0, 'auc': 0, 'win_rate': 0, 'pf': 0,
                'sharpe': 0, 'max_dd': 0, 'calmar': 0, 'mean_ret': 0}

    auc = roc_auc_score(lab, probs[mask]) if len(np.unique(lab)) > 1 else 0.5

    wins  = r[r > 0]
    loses = r[r <= 0]
    win_rate = float(np.mean(lab))
    pf       = float(wins.sum() / (-loses.sum())) if loses.sum() < 0 else float('inf')

    # Equity curve (cumulative returns)
    equity = np.cumprod(1 + r) - 1
    rolling_max = np.maximum.accumulate(equity + 1)
    dd = (equity + 1) / rolling_max - 1
    max_dd = float(dd.min())

    # Sharpe (annualised, assuming ~250 signals/yr as trading-day proxy)
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0

    # Calmar = total_return / abs(max_dd)
    total_ret = float((np.prod(1 + r) - 1))
    calmar    = float(total_ret / abs(max_dd)) if max_dd < 0 else float('inf')

    return {
        'n':         int(n),
        'auc':       round(auc, 4),
        'win_rate':  round(win_rate, 4),
        'pf':        round(min(pf, 99.0), 3),
        'sharpe':    round(sharpe, 3),
        'max_dd':    round(max_dd, 4),
        'calmar':    round(calmar, 3),
        'mean_ret':  round(float(r.mean()), 5),
        'total_ret': round(total_ret, 4),
    }


def train_lgbm_fold(X_tr, y_tr, X_te):
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    p = {**LGBM_PARAMS, 'scale_pos_weight': pos_weight}
    dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    dval   = lgb.Dataset(X_te, label=np.zeros(len(X_te)), reference=dtrain, free_raw_data=False)
    model  = lgb.train(p, dtrain, num_boost_round=300,
                       valid_sets=[dval],
                       callbacks=[lgb.early_stopping(30, verbose=False),
                                  lgb.log_evaluation(-1)])
    return model


# ── 1. Static OOS ─────────────────────────────────────────────────────────────

def static_oos(df, min_prob):
    print("\n== Static OOS (trained model, last 20% of data) ==")
    model_path = os.path.join(MODELS_DIR, 'lgbm_model.txt')
    if not os.path.exists(model_path):
        print("  SKIP: lgbm_model.txt not found (run Phase 1 first).")
        return None

    model   = lgb.Booster(model_file=model_path)
    split   = int(len(df) * 0.8)
    df_test = df.iloc[split:].copy()

    X_te  = df_test[FEATURE_COLS].values.astype(np.float32)
    y_te  = df_test[WIN_COL].values.astype(int)
    ret   = df_test['actual_return'].values.astype(float)
    probs = model.predict(X_te)

    m = compute_metrics(ret, y_te, probs, min_prob=0.0)
    m_filtered = compute_metrics(ret, y_te, probs, min_prob=min_prob)

    print(f"  Test signals   : {len(df_test):,}")
    print(f"  Date range     : {df_test['date'].min()} to {df_test['date'].max()}")
    print(f"  AUC            : {m['auc']:.4f}")
    print(f"  Win Rate       : {m['win_rate']:.1%}  (all)  |  {m_filtered['win_rate']:.1%}  (prob>={min_prob})")
    print(f"  Profit Factor  : {m['pf']:.2f}  (all)  |  {m_filtered['pf']:.2f}  (prob>={min_prob})")
    print(f"  Sharpe         : {m['sharpe']:.3f}")
    print(f"  Max Drawdown   : {m['max_dd']:.2%}")
    print(f"  Calmar         : {m['calmar']:.3f}")
    print(f"  Total Return   : {m['total_ret']:.2%}")

    return {'all': m, 'filtered': m_filtered,
            'date_range': [str(df_test['date'].min()), str(df_test['date'].max())],
            'n_test': len(df_test)}


# ── 2. Walk-Forward ───────────────────────────────────────────────────────────

def walk_forward(df, n_folds, min_prob):
    print(f"\n== Walk-Forward ({n_folds} expanding folds) ==")

    df = df.sort_values('date').reset_index(drop=True)
    fold_size = len(df) // (n_folds + 1)   # last n_folds = test folds; 1st = burn-in

    results = []
    all_oos_ret   = []
    all_oos_lab   = []
    all_oos_prob  = []

    for fold in range(n_folds):
        train_end  = fold_size * (fold + 1)
        test_start = train_end
        test_end   = min(test_start + fold_size, len(df))

        df_tr = df.iloc[:train_end]
        df_te = df.iloc[test_start:test_end]

        if len(df_te) < 100:
            print(f"  Fold {fold+1}: too few test samples, skip.")
            continue

        X_tr = df_tr[FEATURE_COLS].values.astype(np.float32)
        y_tr = df_tr[WIN_COL].values.astype(int)
        X_te = df_te[FEATURE_COLS].values.astype(np.float32)
        y_te = df_te[WIN_COL].values.astype(int)
        ret  = df_te['actual_return'].values.astype(float)

        model = train_lgbm_fold(X_tr, y_tr, X_te)
        # Retrain on full train set without early stopping for final fold probs
        probs = model.predict(X_te)

        m = compute_metrics(ret, y_te, probs, min_prob=min_prob)
        date_lo = str(df_te['date'].min())
        date_hi = str(df_te['date'].max())

        print(f"  Fold {fold+1}/{n_folds}  [{date_lo} - {date_hi}]  "
              f"n={len(df_te):,}  AUC={m['auc']:.4f}  "
              f"WinRate={m['win_rate']:.1%}  PF={m['pf']:.2f}  "
              f"Sharpe={m['sharpe']:.3f}  MaxDD={m['max_dd']:.2%}")

        results.append({**m, 'fold': fold + 1,
                        'train_n': len(df_tr),
                        'date_range': [date_lo, date_hi]})

        all_oos_ret.extend(ret.tolist())
        all_oos_lab.extend(y_te.tolist())
        all_oos_prob.extend(probs.tolist())

    # Aggregate OOS
    agg = compute_metrics(
        np.array(all_oos_ret), np.array(all_oos_lab),
        np.array(all_oos_prob), min_prob=min_prob)

    print(f"\n  -- Aggregate OOS (all {n_folds} folds combined) --")
    print(f"  Total OOS signals : {agg['n']:,}")
    print(f"  AUC               : {agg['auc']:.4f}")
    print(f"  Win Rate          : {agg['win_rate']:.1%}")
    print(f"  Profit Factor     : {agg['pf']:.2f}")
    print(f"  Sharpe            : {agg['sharpe']:.3f}")
    print(f"  Max Drawdown      : {agg['max_dd']:.2%}")
    print(f"  Calmar            : {agg['calmar']:.3f}")
    print(f"  Total OOS Return  : {agg['total_ret']:.2%}")

    # Consistency check: did AUC degrade sharply across folds?
    aucs = [r['auc'] for r in results]
    auc_drift = max(aucs) - min(aucs)
    if auc_drift > 0.05:
        print(f"\n  WARN: AUC drift {auc_drift:.4f} across folds -- possible regime shift or overfit.")
    else:
        print(f"\n  OK: AUC drift {auc_drift:.4f} across folds -- stable.")

    return {'folds': results, 'aggregate': agg, 'auc_drift': round(auc_drift, 4)}


# ── 3. HTML report ────────────────────────────────────────────────────────────

def make_html(static, wf, n_folds, min_prob):
    folds_rows = ''
    for f in wf['folds']:
        dd_col = 'style="color:#e74c3c"' if f['max_dd'] < -0.15 else ''
        auc_col = 'style="color:#27ae60"' if f['auc'] >= 0.57 else ('style="color:#e74c3c"' if f['auc'] < 0.52 else '')
        folds_rows += f"""
        <tr>
          <td>{f['fold']}</td>
          <td>{f['date_range'][0]} – {f['date_range'][1]}</td>
          <td>{f['train_n']:,}</td>
          <td>{f['n']:,}</td>
          <td {auc_col}>{f['auc']:.4f}</td>
          <td>{f['win_rate']:.1%}</td>
          <td>{f['pf']:.2f}</td>
          <td>{f['sharpe']:.3f}</td>
          <td {dd_col}>{f['max_dd']:.2%}</td>
          <td>{f['calmar']:.3f}</td>
          <td>{f['total_ret']:.2%}</td>
        </tr>"""

    agg = wf['aggregate']
    oos = static or {}
    oos_all = oos.get('all', {})
    oos_flt = oos.get('filtered', {})

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>OOS Backtest Report</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background:#0d1117; color:#c9d1d9; margin:0; padding:24px; }}
  h1 {{ color:#58a6ff; font-size:1.5rem; margin-bottom:4px; }}
  h2 {{ color:#8b949e; font-size:1rem; border-bottom:1px solid #21262d; padding-bottom:6px; margin-top:28px; }}
  .meta {{ color:#6e7681; font-size:0.82rem; margin-bottom:20px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin:16px 0; }}
  .card {{ background:#161b22; border:1px solid #21262d; border-radius:8px; padding:16px; }}
  .card .label {{ font-size:0.75rem; color:#6e7681; margin-bottom:4px; }}
  .card .value {{ font-size:1.4rem; font-weight:700; color:#e6edf3; }}
  .card .value.good {{ color:#3fb950; }}
  .card .value.warn {{ color:#f0883e; }}
  .card .value.bad  {{ color:#f85149; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.85rem; margin-top:12px; }}
  th {{ background:#21262d; padding:8px 10px; text-align:left; color:#8b949e; font-weight:600; }}
  td {{ padding:7px 10px; border-bottom:1px solid #21262d; }}
  tr:hover td {{ background:#1c2128; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; }}
  .badge.ok  {{ background:#1a4731; color:#3fb950; }}
  .badge.warn {{ background:#3d2b00; color:#f0883e; }}
  .badge.bad  {{ background:#3d1111; color:#f85149; }}
  .section {{ background:#161b22; border:1px solid #21262d; border-radius:10px; padding:20px; margin-bottom:20px; }}
</style>
</head>
<body>
<h1>Dr KKR CPR Screener — OOS Backtest Report</h1>
<div class="meta">WIN_COL={WIN_COL} &nbsp;|&nbsp; min_prob={min_prob} &nbsp;|&nbsp; {n_folds}-fold walk-forward &nbsp;|&nbsp; FEATURE_COLS={len(FEATURE_COLS)}</div>

<div class="section">
<h2>Static OOS (trained model, last 20% of data)</h2>
{"<em>lgbm_model.txt not found — run Phase 1 first.</em>" if not static else f'''
<div class="grid">
  <div class="card"><div class="label">OOS Signals</div><div class="value">{oos.get("n_test","--"):,}</div></div>
  <div class="card"><div class="label">AUC</div><div class="value {"good" if oos_all.get("auc",0)>=0.57 else "warn" if oos_all.get("auc",0)>=0.52 else "bad"}">{oos_all.get("auc","--")}</div></div>
  <div class="card"><div class="label">Win Rate (all)</div><div class="value">{oos_all.get("win_rate",0):.1%}</div></div>
  <div class="card"><div class="label">Win Rate (filtered)</div><div class="value good">{oos_flt.get("win_rate",0):.1%}</div></div>
  <div class="card"><div class="label">Profit Factor</div><div class="value {"good" if oos_all.get("pf",0)>=1.3 else "warn"}">{oos_all.get("pf","--")}</div></div>
  <div class="card"><div class="label">Sharpe</div><div class="value {"good" if oos_all.get("sharpe",0)>=1 else "warn"}">{oos_all.get("sharpe","--")}</div></div>
  <div class="card"><div class="label">Max Drawdown</div><div class="value {"bad" if oos_all.get("max_dd",0)<-0.2 else "warn" if oos_all.get("max_dd",0)<-0.1 else "good"}">{oos_all.get("max_dd",0):.2%}</div></div>
  <div class="card"><div class="label">Total Return</div><div class="value {"good" if oos_all.get("total_ret",0)>0 else "bad"}">{oos_all.get("total_ret",0):.2%}</div></div>
</div>
<div class="meta">Date range: {oos.get("date_range",["--","--"])[0]} to {oos.get("date_range",["--","--"])[1]}</div>
'''}
</div>

<div class="section">
<h2>Walk-Forward — Fold Results</h2>
<table>
<tr><th>Fold</th><th>Test Period</th><th>Train N</th><th>Test N</th><th>AUC</th><th>Win Rate</th><th>PF</th><th>Sharpe</th><th>MaxDD</th><th>Calmar</th><th>OOS Return</th></tr>
{folds_rows}
</table>
</div>

<div class="section">
<h2>Walk-Forward — Aggregate OOS</h2>
<div class="grid">
  <div class="card"><div class="label">Total OOS Signals</div><div class="value">{agg["n"]:,}</div></div>
  <div class="card"><div class="label">AUC</div><div class="value {"good" if agg["auc"]>=0.57 else "warn" if agg["auc"]>=0.52 else "bad"}">{agg["auc"]}</div></div>
  <div class="card"><div class="label">Win Rate</div><div class="value">{agg["win_rate"]:.1%}</div></div>
  <div class="card"><div class="label">Profit Factor</div><div class="value {"good" if agg["pf"]>=1.3 else "warn"}">{agg["pf"]}</div></div>
  <div class="card"><div class="label">Sharpe</div><div class="value {"good" if agg["sharpe"]>=1 else "warn"}">{agg["sharpe"]}</div></div>
  <div class="card"><div class="label">Max Drawdown</div><div class="value {"bad" if agg["max_dd"]<-0.2 else "warn" if agg["max_dd"]<-0.1 else "good"}">{agg["max_dd"]:.2%}</div></div>
  <div class="card"><div class="label">Calmar</div><div class="value {"good" if agg["calmar"]>=1 else "warn"}">{agg["calmar"]}</div></div>
  <div class="card"><div class="label">OOS Total Return</div><div class="value {"good" if agg["total_ret"]>0 else "bad"}">{agg["total_ret"]:.2%}</div></div>
</div>
<p>AUC drift across folds: <strong>{wf["auc_drift"]:.4f}</strong> &nbsp;
   <span class="badge {"ok" if wf["auc_drift"]<=0.05 else "warn" if wf["auc_drift"]<=0.10 else "bad"}">
   {"STABLE" if wf["auc_drift"]<=0.05 else "MODERATE DRIFT" if wf["auc_drift"]<=0.10 else "HIGH DRIFT - CHECK OVERFIT"}</span>
</p>
</div>

<div class="meta" style="margin-top:32px;">Generated by backtest_oos.py &nbsp;|&nbsp; Dr KKR CPR Screener ML Pipeline</div>
</body>
</html>"""
    return html


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds',    type=int,   default=5,   help='Walk-forward folds (default 5)')
    parser.add_argument('--min-prob', type=float, default=0.5, help='Min LightGBM prob to trade (default 0.5)')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  OOS + Walk-Forward Backtest")
    print(f"  WIN_COL={WIN_COL}  FEATURES={len(FEATURE_COLS)}  FOLDS={args.folds}  MIN_PROB={args.min_prob}")
    print("="*60)

    df = load_signal_dataset()
    if WIN_COL not in df.columns:
        raise KeyError(f"WIN_COL='{WIN_COL}' not in dataset — rebuild with build_dataset.py")
    if 'actual_return' not in df.columns:
        raise KeyError("'actual_return' not in dataset — rebuild with build_dataset.py")

    df = df.sort_values('date').reset_index(drop=True)
    print(f"  Dataset: {len(df):,} signals  |  {df['date'].min()} to {df['date'].max()}")
    print(f"  Target win rate (raw): {df[WIN_COL].mean():.1%}")

    # Fill missing new feature cols with neutral defaults
    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"  WARN: '{col}' missing in dataset — filling with 0.0")
            df[col] = 0.0

    static = static_oos(df, args.min_prob)
    wf     = walk_forward(df, args.folds, args.min_prob)

    # Save JSON
    results = {
        'config': {
            'WIN_COL':    WIN_COL,
            'n_features': len(FEATURE_COLS),
            'n_folds':    args.folds,
            'min_prob':   args.min_prob,
        },
        'static_oos':    static,
        'walk_forward':  wf,
    }
    json_path = os.path.join(MODELS_DIR, 'backtest_oos_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {json_path}")

    # Save HTML
    html = make_html(static, wf, args.folds, args.min_prob)
    html_path = os.path.join(MODELS_DIR, 'backtest_oos_report.html')
    with open(html_path, 'w') as f:
        f.write(html)
    print(f"  HTML report  -> {html_path}")

    print("\n" + "="*60)
    print("  DONE")
    print("="*60)


if __name__ == '__main__':
    main()
