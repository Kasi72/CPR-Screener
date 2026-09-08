"""
backfill_hmm_posteriors.py — Extend HMM posteriors back to 2018-01-01.

Fixes the regime=-1 gap caused by train_phase1.py using only 5y of Nifty data
while signal_dataset.csv has signals going back further.

Refits GaussianHMM on the full available Nifty history (2018-01-01 to today),
overwrites hmm_posteriors.json and hmm_params.json with extended coverage,
then re-runs build_dataset.py so all signals get a valid hmm_regime.

Usage:
    python scripts/ml/backfill_hmm_posteriors.py
    python scripts/ml/backfill_hmm_posteriors.py --skip-rebuild
"""

import os, sys, json, argparse, subprocess, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.ml.data_utils import MODELS_DIR, build_nifty_regime_features

REGIME_NAMES = ['Bull-Trend', 'Bear-Trend', 'Chop', 'High-Vol-Panic']
BASE         = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fit_hmm_extended(start='2018-01-01'):
    print(f"\n── Backfill HMM Posteriors (start={start}) ─────────────────────")
    print("  Downloading ^NSEI ...")
    nifty = yf.download('^NSEI', start=start, interval='1d', progress=False, auto_adjust=True)
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    if nifty.empty:
        raise RuntimeError("Failed to download Nifty 50 data.")
    print(f"  Got {len(nifty)} daily bars ({nifty.index[0].date()} to {nifty.index[-1].date()})")

    obs = build_nifty_regime_features(nifty)
    print(f"  Observation matrix: {obs.shape}")

    scaler = StandardScaler().fit(obs)
    obs_sc = scaler.transform(obs)

    # Refit HMM (same hyperparams as train_phase1.py)
    best_model, best_score = None, -np.inf
    for seed in [42, 7, 99]:
        model = GaussianHMM(
            n_components=4,
            covariance_type='diag',
            n_iter=200,
            random_state=seed,
            tol=1e-4
        )
        model.fit(obs_sc)
        score = model.score(obs_sc)
        print(f"  seed={seed}  log-likelihood={score:.2f}")
        if score > best_score:
            best_score, best_model = score, model

    states = best_model.predict(obs_sc)
    counts = np.bincount(states, minlength=4)
    print(f"  State distribution: {dict(enumerate(counts.tolist()))}")

    # Map states to regime names by mean return (same logic as train_phase1.py)
    mean_rets  = [obs[states == s, 0].mean() for s in range(4)]
    order      = np.argsort(mean_rets)[::-1]
    regime_map = {
        int(order[0]): 'Bull-Trend',
        int(order[1]): 'Bear-Trend',
        int(order[2]): 'Chop',
        int(order[3]): 'High-Vol-Panic',
    }

    # Date alignment: obs has 200 rows dropped at start
    hmm_dates = nifty.index[1:]   # diff removes one row
    offset    = 200
    hmm_dates = hmm_dates[offset:]
    assert len(hmm_dates) == len(states), \
        f"Length mismatch: {len(hmm_dates)} dates vs {len(states)} states"

    current_state  = int(states[-1])
    current_regime = regime_map[current_state]
    print(f"  Current regime: {current_regime} (state {current_state})")

    # Posteriors [T, 4]
    posteriors = best_model.predict_proba(obs_sc)
    posterior_by_date = {
        pd.Timestamp(d).strftime('%Y-%m-%d'): posteriors[i].tolist()
        for i, d in enumerate(hmm_dates)
    }
    state_by_date = {
        pd.Timestamp(d).strftime('%Y-%m-%d'): int(s)
        for d, s in zip(hmm_dates, states)
    }

    print(f"  Posteriors cover {len(posterior_by_date)} dates "
          f"({min(posterior_by_date)} to {max(posterior_by_date)})")

    # Regime stability + transition risk per date (posterior-based, matches build_dataset.py)
    # regime_stability = max(posterior_probs)  range [0.25, 1.0] for 4 states
    # transition_risk  = normalized entropy H/log(4)  range [0=certain, 1=max uncertainty]
    _log_n = np.log(4)
    regime_stability_by_date = {}
    transition_risk_by_date  = {}
    for i, date_str in enumerate(state_by_date.keys()):
        post = posteriors[i]
        stability = float(post.max())
        entropy   = float(-np.sum(post * np.log(np.clip(post, 1e-9, 1))) / _log_n)
        regime_stability_by_date[date_str] = round(stability, 6)
        transition_risk_by_date[date_str]  = round(entropy,   6)

    # Build hmm_params.json (compatible with JS Viterbi + predict_server.py)
    params = {
        'n_components':  4,
        'startprob':     best_model.startprob_.tolist(),
        'transmat':      best_model.transmat_.tolist(),
        'means':         best_model.means_.tolist(),
        'covars':        best_model.covars_.tolist(),
        'regime_map':    {str(k): v for k, v in regime_map.items()},
        'current_state': current_state,
        'current_regime': current_regime,
        'state_by_date': state_by_date,
        'regime_stability_by_date': regime_stability_by_date,
        'transition_risk_by_date':  transition_risk_by_date,
    }

    # Save
    post_out  = os.path.join(MODELS_DIR, 'hmm_posteriors.json')
    param_out = os.path.join(MODELS_DIR, 'hmm_params.json')

    with open(post_out, 'w') as f:
        json.dump(posterior_by_date, f)
    print(f"  Saved → {post_out}  ({len(posterior_by_date)} dates)")

    with open(param_out, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"  Saved → {param_out}")

    return len(posterior_by_date)


def rebuild_dataset():
    print("\n── Rebuilding signal_dataset.csv with extended regime coverage ──")
    script = os.path.join(BASE, 'scripts', 'ml', 'build_dataset.py')
    ret    = subprocess.run([sys.executable, script], cwd=BASE)
    if ret.returncode != 0:
        print("  build_dataset.py failed — check output above.")
        sys.exit(1)
    print("  build_dataset.py complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-rebuild', action='store_true',
                        help='Only backfill posteriors, skip re-running build_dataset.py')
    parser.add_argument('--start', default='2018-01-01',
                        help='Start date for Nifty download (default: 2018-01-01)')
    args = parser.parse_args()

    n_dates = fit_hmm_extended(start=args.start)

    if not args.skip_rebuild:
        rebuild_dataset()
        print(f"\nDone. HMM posteriors extended to {n_dates} dates.")
        print("signal_dataset.csv rebuilt — regime=-1 gap should be <5%.")
    else:
        print(f"\nDone. HMM posteriors extended to {n_dates} dates.")
        print("Run build_dataset.py manually to apply to signal_dataset.csv.")


if __name__ == '__main__':
    main()
