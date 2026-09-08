"""
kaggle_phase2c_runner.py — Phase 2c Regime-Conditional LightGBM HPO on Kaggle.

Uploads signal_dataset.csv (must have Sprint 1 CPR columns + hmm_regime)
to drkasi/cpr-screener-phase2c-inputs, then runs the kernel.

Flow:
  1. Upload dataset — drkasi/cpr-screener-phase2c-inputs (dedicated)
  2. Wait for ready — poll kaggle datasets files
  3. Push kernel    — drkasi/cpr-phase-2c-lgbm-signal-scorer
  4. Poll status    — wait for complete (default 120 min, regime models take longer)
  5. Download       — lgbm2c_global.txt, lgbm2c_regime_*.txt,
                      shap_weights2c.json, phase2c_metrics.json

Usage:
    python scripts/ml/kaggle_phase2c_runner.py
    python scripts/ml/kaggle_phase2c_runner.py --no-upload
    python scripts/ml/kaggle_phase2c_runner.py --timeout-minutes 150
"""

import os, sys, json, time, shutil, subprocess, tempfile, glob, zipfile, argparse

BASE       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KAGGLE_DIR = os.path.join(BASE, 'kaggle', 'phase2c')
MODELS_DIR = os.path.join(BASE, 'models')
SIGNAL_CSV = os.path.join(MODELS_DIR, 'signal_dataset.csv')

DATASET_SLUG = 'drkasi/cpr-screener-phase2c-inputs'
KERNEL_SLUG  = 'drkasi/cpr-phase-2c-lgbm-signal-scorer'


def _run(args, check=True, capture=False):
    r = subprocess.run(args, capture_output=True, text=True)
    if not capture:
        if r.stdout:
            print(r.stdout, end='', flush=True)
        if r.stderr:
            print(r.stderr, end='', flush=True)
    if check and r.returncode != 0:
        msg = (r.stderr or r.stdout or '').strip()
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(args)}\n{msg}")
    return r


def _kaggle(*args, capture=False):
    return _run(['kaggle'] + list(args), capture=capture)


def upload_dataset():
    print('  [1/5] Uploading signal_dataset.csv to Phase 2c dataset ...')
    if not os.path.exists(SIGNAL_CSV):
        raise FileNotFoundError(
            f'signal_dataset.csv not found at {SIGNAL_CSV}\n'
            f'Run build_dataset.py (Sprint 1) first to generate the Sprint 1 columns.'
        )

    # Validate Sprint 1 columns are present
    import pandas as pd
    sample = pd.read_csv(SIGNAL_CSV, nrows=5)
    required = ['cpr_overlap_pct', 'open_to_cpr_dist', 'prev_cpr_respected',
                'cpr_zone_vol_ratio', 'hmm_regime', 'hit_t3',
                'open_inside_cpr', 'cpr_virgin', 'consecutive_narrow_cprs',
                'cpr_midpoint_trend', 'cpr_expansion_factor',
                # Sprint 3: gap + bar quality + volatility + volume structure
                'gap_pct', 'cpr_test_count_5d', 'prev_bar_close_pos',
                'atr_expansion', 'vol_trend_slope',
                # Sprint 4: weekly CPR
                'weekly_cpr_first_break', 'weekly_price_above_wtc']
    missing = [c for c in required if c not in sample.columns]
    if missing:
        raise ValueError(
            f'signal_dataset.csv is missing columns: {missing}\n'
            f'Run build_dataset.py first to regenerate with Sprint 3/4 columns.'
        )
    print(f'    All feature columns verified ({len(required)} checked).')

    staging = tempfile.mkdtemp(prefix='kaggle_p2c_stage_')
    try:
        meta = {
            'title':    'CPR Screener Phase 2c Inputs',
            'id':       DATASET_SLUG,
            'licenses': [{'name': 'other'}],
        }
        with open(os.path.join(staging, 'dataset-metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        size_mb = os.path.getsize(SIGNAL_CSV) / 1024 ** 2
        print(f'    Copying signal_dataset.csv ({size_mb:.0f} MB) ...')
        shutil.copy2(SIGNAL_CSV, os.path.join(staging, 'signal_dataset.csv'))

        try:
            _kaggle('datasets', 'version', '-p', staging,
                    '-m', f'Phase2c signal_dataset {time.strftime("%Y-%m-%d %H:%M")}')
            print('    Dataset version created.')
        except RuntimeError:
            print('    Dataset not found — creating new dataset ...')
            _kaggle('datasets', 'create', '-p', staging)

        print('    Upload submitted. Polling until dataset ready ...')
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def wait_for_dataset_ready(max_wait_s=600):
    deadline = time.time() + max_wait_s
    attempt  = 0
    while time.time() < deadline:
        attempt += 1
        r = _run(['kaggle', 'datasets', 'files', DATASET_SLUG], check=False, capture=True)
        out = (r.stdout or '') + (r.stderr or '')
        if 'signal_dataset.csv' in out:
            print(f'    Dataset ready (attempt {attempt}). Waiting 300s for version commit ...')
            time.sleep(300)
            return
        print(f'    [{attempt}] Not ready yet. Waiting 30s ...')
        time.sleep(30)
    print(f'    Warning: dataset not confirmed ready after {max_wait_s}s — proceeding anyway.')


def push_kernel():
    print('  [3/5] Pushing Phase 2c kernel ...')
    _kaggle('kernels', 'push', '-p', KAGGLE_DIR)
    print('    Kernel pushed. Regime-conditional HPO started on Kaggle.')


def poll_kernel(timeout_minutes=120):
    print(f'  [4/5] Polling kernel status (timeout: {timeout_minutes} min) ...')
    deadline = time.time() + timeout_minutes * 60
    start    = time.time()

    while time.time() < deadline:
        r   = _run(['kaggle', 'kernels', 'status', KERNEL_SLUG], check=False, capture=True)
        out = ((r.stdout or '') + (r.stderr or '')).strip().lower()

        if 'complete' in out:
            status = 'complete'
        elif 'error' in out:
            status = 'error'
        elif 'running' in out:
            status = 'running'
        elif 'queued' in out or 'pending' in out:
            status = 'queued'
        elif 'cancelacknowledged' in out or 'cancelled' in out:
            status = 'cancelacknowledged'
        else:
            status = 'unknown'

        elapsed = (time.time() - start) / 60
        print(f'    [{elapsed:5.1f} min]  status: {status}', flush=True)

        if status == 'complete':
            print('    Kernel complete!')
            return True
        if status in ('error', 'cancelacknowledged'):
            print(f'    Kernel failed: {status}')
            return False

        time.sleep(60)

    print(f'    Timeout after {timeout_minutes} minutes.')
    return False


def download_outputs():
    print('  [5/5] Downloading Phase 2c outputs ...')
    out_dir = tempfile.mkdtemp(prefix='kaggle_p2c_out_')
    try:
        _kaggle('kernels', 'output', KERNEL_SLUG, '-p', out_dir)

        zips = glob.glob(os.path.join(out_dir, '*.zip'))
        if zips:
            print(f'    Extracting {os.path.basename(zips[0])} ...')
            with zipfile.ZipFile(zips[0], 'r') as zf:
                zf.extractall(out_dir)

        # Required output files
        target_files = [
            'lgbm2c_global.txt',
            'shap_weights2c.json',
            'phase2c_metrics.json',
        ]
        # Optional per-regime models
        for i in range(4):
            target_files.append(f'lgbm2c_regime_{i}.txt')

        required = {'lgbm2c_global.txt', 'shap_weights2c.json', 'phase2c_metrics.json'}
        copied = []
        for fn in target_files:
            matches = glob.glob(os.path.join(out_dir, '**', fn), recursive=True)
            if matches:
                shutil.copy2(matches[0], os.path.join(MODELS_DIR, fn))
                copied.append(fn)
                print(f'    OK {fn}')
            elif fn in required:
                print(f'    MISS {fn} (required — kernel may have failed)')
            else:
                print(f'    MISS {fn} (optional — regime may have been skipped)')

        missing_required = required - set(copied)
        if missing_required:
            raise RuntimeError(f'Missing required output files: {missing_required}')

        print(f'    {len(copied)} files saved to {MODELS_DIR}')

        # Print metrics summary
        metrics_path = os.path.join(MODELS_DIR, 'phase2c_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            print('\n  Phase 2c Results:')
            print(f'    Target  : {m.get("target", "?")}')
            print(f'    Features: {m.get("n_features", "?")}')
            print(f'    Val AUC (global): {m.get("final_val_auc_global", "?"):.4f}')
            print(f'    Test AUC(global): {m.get("test_auc_global", "?"):.4f}')
            print(f'    Runtime : {m.get("runtime_min", "?")} min')
            for r_id, r_stats in m.get('regimes', {}).items():
                status = r_stats.get('status', '?')
                if status == 'trained':
                    print(f'    Regime {r_id}: val={r_stats["val_auc"]:.4f}  '
                          f'test={r_stats["test_auc"]:.4f}  n={r_stats["n_train"]:,}')
                else:
                    print(f'    Regime {r_id}: {status} (n={r_stats["n_train"]:,})')

        print('\n  Phase 2c download done.')
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def main(timeout_minutes=120, upload=True):
    print('\n' + '=' * 60)
    print('  Phase 2c: Regime-Conditional LightGBM HPO Runner')
    print('=' * 60)

    if not os.path.exists(SIGNAL_CSV):
        print('  WARN: signal_dataset.csv not found — skipping Phase 2c (Kaggle HPO)')
        print('  Existing lgbm2c models retained.')
        return

    if upload:
        upload_dataset()
        wait_for_dataset_ready()
    else:
        print('  [1/5] Skipping dataset upload (--no-upload flag set).')
        print('  [2/5] Skipping dataset ready check.')

    push_kernel()
    ok = poll_kernel(timeout_minutes=timeout_minutes)
    if not ok:
        sys.exit(1)
    download_outputs()

    print('\n  Phase 2c complete. Models saved to:', MODELS_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout-minutes', type=int, default=120,
                        help='Max minutes to wait for Kaggle kernel (default 120)')
    parser.add_argument('--no-upload', action='store_true',
                        help='Skip dataset upload (reuse existing Kaggle dataset)')
    args = parser.parse_args()
    main(timeout_minutes=args.timeout_minutes, upload=not args.no_upload)
