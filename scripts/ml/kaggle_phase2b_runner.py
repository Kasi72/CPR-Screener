"""
kaggle_phase2b_runner.py — Phase 2b LightGBM HPO Signal Scorer on Kaggle CPU.

Uploads signal_dataset.csv to its own dedicated dataset
(drkasi/cpr-screener-phase2b-inputs), then runs the kernel.

Flow:
  1. Upload dataset — drkasi/cpr-screener-phase2b-inputs (dedicated)
  2. Wait for ready — poll kaggle datasets files
  3. Push kernel    — drkasi/cpr-phase-2b-lightgbm-hpo-signal-scorer
  4. Poll status    — wait for complete (default 90 min)
  5. Download       — lgbm_scorer.txt, shap_weights.json, phase2b_metrics.json

Usage:
    python scripts/ml/kaggle_phase2b_runner.py
    python scripts/ml/kaggle_phase2b_runner.py --no-upload
    python scripts/ml/kaggle_phase2b_runner.py --timeout-minutes 120
"""

import os, sys, json, time, shutil, subprocess, tempfile, glob, zipfile, argparse

BASE       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KAGGLE_DIR = os.path.join(BASE, 'kaggle', 'phase2b')
MODELS_DIR = os.path.join(BASE, 'models')
SIGNAL_CSV = os.path.join(MODELS_DIR, 'signal_dataset.csv')

DATASET_SLUG = 'drkasi/cpr-screener-phase2b-inputs'
KERNEL_SLUG  = 'drkasi/cpr-phase-2b-lightgbm-hpo-signal-scorer'


def _run(args, check=True, capture=False):
    r = subprocess.run(args, capture_output=capture, text=True)
    if check and r.returncode != 0:
        msg = (r.stderr or r.stdout or '').strip()
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(args)}\n{msg}")
    return r


def _kaggle(*args, capture=False):
    return _run(['kaggle'] + list(args), capture=capture)


def upload_dataset():
    print('  [1/5] Uploading signal_dataset.csv to Kaggle ...')
    if not os.path.exists(SIGNAL_CSV):
        raise FileNotFoundError(f'signal_dataset.csv not found at {SIGNAL_CSV}')

    staging = tempfile.mkdtemp(prefix='kaggle_p2b_stage_')
    try:
        meta = {
            'title': 'CPR Screener Phase 2b Inputs',
            'id':    DATASET_SLUG,
            'licenses': [{'name': 'other'}],
        }
        with open(os.path.join(staging, 'dataset-metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        size_mb = os.path.getsize(SIGNAL_CSV) / 1024 ** 2
        print(f'    Copying signal_dataset.csv ({size_mb:.0f} MB) ...')
        shutil.copy2(SIGNAL_CSV, os.path.join(staging, 'signal_dataset.csv'))

        r = _kaggle('datasets', 'list', '--search', 'cpr-screener-phase2b-inputs',
                    '--user', 'drkasi', capture=True)
        exists = 'cpr-screener-phase2b-inputs' in (r.stdout or '')

        if exists:
            _kaggle('datasets', 'version', '-p', staging,
                    '-m', f'Phase2b signal_dataset {time.strftime("%Y-%m-%d %H:%M")}')
        else:
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
            print(f'    Dataset ready (attempt {attempt}).')
            return
        print(f'    [{attempt}] Not ready yet. Waiting 30s ...')
        time.sleep(30)
    print(f'    Warning: dataset not confirmed ready after {max_wait_s}s — proceeding anyway.')


def push_kernel():
    print('  [3/5] Pushing Phase 2b kernel ...')
    _kaggle('kernels', 'push', '-p', KAGGLE_DIR)
    print('    Kernel pushed. LightGBM HPO started on Kaggle.')


def poll_kernel(timeout_minutes=90):
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
    print('  [5/5] Downloading Phase 2b outputs ...')
    out_dir = tempfile.mkdtemp(prefix='kaggle_p2b_out_')
    try:
        _kaggle('kernels', 'output', KERNEL_SLUG, '-p', out_dir)

        zips = glob.glob(os.path.join(out_dir, '*.zip'))
        if zips:
            print(f'    Extracting {os.path.basename(zips[0])} ...')
            with zipfile.ZipFile(zips[0], 'r') as zf:
                zf.extractall(out_dir)

        target_files = ['lgbm_scorer.txt', 'shap_weights.json', 'phase2b_metrics.json']
        copied = []
        for fn in target_files:
            matches = glob.glob(os.path.join(out_dir, '**', fn), recursive=True)
            if matches:
                shutil.copy2(matches[0], os.path.join(MODELS_DIR, fn))
                copied.append(fn)
                print(f'    ✓ {fn}')
            else:
                print(f'    ✗ {fn} NOT FOUND in kernel output')

        if len(copied) < len(target_files):
            missing = set(target_files) - set(copied)
            raise RuntimeError(f'Missing output files: {missing}')

        print(f'    All {len(copied)} output files saved to {MODELS_DIR}')

        # Print metrics summary
        metrics_path = os.path.join(MODELS_DIR, 'phase2b_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            print(f'\n  Phase 2b Results:')
            print(f'    CV AUC  : {m.get("best_cv_auc", "?"):.4f}')
            print(f'    Val AUC : {m.get("final_val_auc", "?"):.4f}')
            print(f'    Runtime : {m.get("runtime_min", "?")} min')

    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def main(timeout_minutes=90, upload=True):
    print('\n' + '=' * 60)
    print('  Phase 2b: Kaggle LightGBM HPO Runner')
    print('=' * 60)

    if not os.path.exists(SIGNAL_CSV):
        print('  WARN: signal_dataset.csv not found — skipping Phase 2b (Kaggle HPO)')
        print('  Existing lgbm_scorer.txt retained.')
        return

    if upload:
        upload_dataset()
        wait_for_dataset_ready()
    else:
        print('  [1/5] Skipping dataset upload (--no-upload flag set).')

    push_kernel()
    ok = poll_kernel(timeout_minutes=timeout_minutes)
    if not ok:
        sys.exit(1)
    download_outputs()

    print('\n  Phase 2b complete. Models saved to:', MODELS_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout-minutes', type=int, default=90)
    parser.add_argument('--no-upload', action='store_true',
                        help='Skip dataset upload (reuse existing Kaggle dataset)')
    args = parser.parse_args()
    main(timeout_minutes=args.timeout_minutes, upload=not args.no_upload)
