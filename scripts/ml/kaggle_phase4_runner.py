"""
kaggle_phase4_runner.py — Orchestrates Phase 4 PPO training on Kaggle.

Reuses the Phase 3 dataset (drkasi/cpr-screener-phase3-inputs) which already
contains signal_dataset.csv — so upload is skipped by default.

Flow:
  1. (Optional) Upload dataset  — skipped by default via --skip-upload
  2. Push kernel                — drkasi/cpr-phase-4-ppo-sizing
  3. Poll status                — wait for complete (60-min timeout)
  4. Download outputs           — ppo_policy.zip, ppo_policy_weights.json, phase4_metrics.json

Usage:
    python scripts/ml/kaggle_phase4_runner.py
    python scripts/ml/kaggle_phase4_runner.py --upload   # force dataset re-upload
    python scripts/ml/kaggle_phase4_runner.py --timeout-minutes 90
"""

import os, sys, json, time, shutil, subprocess, tempfile, glob, zipfile, argparse

BASE        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KAGGLE_DIR  = os.path.join(BASE, 'kaggle', 'phase4')
MODELS_DIR  = os.path.join(BASE, 'models')
SIGNAL_CSV  = os.path.join(MODELS_DIR, 'signal_dataset.csv')

DATASET_SLUG = 'drkasi/cpr-screener-phase4-inputs'   # dedicated Phase 4 dataset
KERNEL_SLUG  = 'drkasi/cpr-phase-4-ppo-position-sizing'


def _run(args, check=True, capture=False):
    r = subprocess.run(args, capture_output=capture, text=True)
    if check and r.returncode != 0:
        msg = (r.stderr or r.stdout or '').strip()
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(args)}\n{msg}")
    return r


def _kaggle(*args, capture=False):
    return _run(['kaggle'] + list(args), capture=capture)


def upload_dataset():
    """Upload signal_dataset.csv to dedicated Phase 4 dataset (no --dir-mode zip)."""
    print("  [1/4] Uploading signal_dataset.csv to Kaggle ...")
    if not os.path.exists(SIGNAL_CSV):
        raise FileNotFoundError(f"signal_dataset.csv not found at {SIGNAL_CSV}")

    staging = tempfile.mkdtemp(prefix='kaggle_p4_stage_')
    try:
        meta = {
            "title": "CPR Screener Phase 4 Inputs",
            "id":    DATASET_SLUG,
            "licenses": [{"name": "other"}]
        }
        with open(os.path.join(staging, 'dataset-metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        size_mb = os.path.getsize(SIGNAL_CSV) / 1024 ** 2
        print(f"    Copying signal_dataset.csv ({size_mb:.0f} MB) ...")
        shutil.copy2(SIGNAL_CSV, os.path.join(staging, 'signal_dataset.csv'))

        r = _kaggle('datasets', 'list', '--search', 'cpr-screener-phase4-inputs',
                    '--user', 'drkasi', capture=True)
        exists = 'cpr-screener-phase4-inputs' in (r.stdout or '')

        if exists:
            _kaggle('datasets', 'version', '-p', staging,
                    '-m', f'Phase4 signal_dataset {time.strftime("%Y-%m-%d %H:%M")}')
        else:
            _kaggle('datasets', 'create', '-p', staging)

        print("    Upload submitted. Polling until dataset ready ...")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def wait_for_dataset_ready(max_wait_s=600):
    """Poll kaggle datasets files until signal_dataset.csv is visible."""
    deadline = time.time() + max_wait_s
    attempt  = 0
    while time.time() < deadline:
        attempt += 1
        r = _run(['kaggle', 'datasets', 'files', DATASET_SLUG], check=False, capture=True)
        out = (r.stdout or '') + (r.stderr or '')
        if 'signal_dataset.csv' in out:
            print(f"    Dataset ready (attempt {attempt}). Waiting 90s for version commit ...")
            time.sleep(90)
            return
        print(f"    [{attempt}] Not ready yet. Waiting 30s ...")
        time.sleep(30)
    print(f"    Warning: dataset not confirmed ready after {max_wait_s}s — proceeding anyway.")


def push_kernel():
    print("  [2/4] Pushing Phase 4 kernel ...")
    _kaggle('kernels', 'push', '-p', KAGGLE_DIR)
    print("    Kernel pushed. PPO training started on Kaggle.")


def poll_kernel(timeout_minutes=60):
    print(f"  [3/4] Polling kernel status (timeout: {timeout_minutes} min) ...")
    deadline = time.time() + timeout_minutes * 60
    start    = time.time()

    while time.time() < deadline:
        r = _run(['kaggle', 'kernels', 'status', KERNEL_SLUG], check=False, capture=True)
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
        print(f"    [{elapsed:5.1f} min]  status: {status}", flush=True)

        if status == 'complete':
            print("    Kernel complete!")
            return True
        if status in ('error', 'cancelacknowledged'):
            print(f"    Kernel failed: {status}")
            return False

        time.sleep(60)

    print(f"    Timeout after {timeout_minutes} minutes.")
    return False


def download_outputs():
    print("  [4/4] Downloading Phase 4 outputs ...")
    out_dir = tempfile.mkdtemp(prefix='kaggle_p4_out_')
    try:
        # Use check=False: kaggle CLI exits 1 on Windows charmap errors even when
        # files download successfully. Verify files exist instead of trusting exit code.
        _run(['kaggle', 'kernels', 'output', KERNEL_SLUG, '-p', out_dir], check=False)

        zips = glob.glob(os.path.join(out_dir, '*.zip'))
        if zips:
            print(f"    Extracting {os.path.basename(zips[0])} ...")
            with zipfile.ZipFile(zips[0], 'r') as zf:
                zf.extractall(out_dir)

        target_files = ['ppo_policy.zip', 'ppo_policy_weights.json', 'phase4_metrics.json']
        copied = []
        for fn in target_files:
            matches = glob.glob(os.path.join(out_dir, '**', fn), recursive=True)
            if matches:
                shutil.copy2(matches[0], os.path.join(MODELS_DIR, fn))
                copied.append(fn)
                print(f"    ✓ {fn}")
            else:
                print(f"    ✗ {fn} NOT FOUND in kernel output")

        if len(copied) < len(target_files):
            missing = set(target_files) - set(copied)
            raise RuntimeError(f"Missing output files: {missing}")

        print(f"    All {len(copied)} output files saved to {MODELS_DIR}")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def main(timeout_minutes=90, upload=True):
    print("\n" + "=" * 60)
    print("  Phase 4: Kaggle PPO Runner")
    print("=" * 60)

    if not os.path.exists(SIGNAL_CSV):
        print('  WARN: signal_dataset.csv not found — skipping Phase 4 (Kaggle PPO)')
        print('  Existing ppo_policy_weights.json retained.')
        return

    if upload:
        upload_dataset()
        wait_for_dataset_ready()
    else:
        print("  [1/4] Skipping dataset upload (--no-upload flag set).")

    push_kernel()
    ok = poll_kernel(timeout_minutes=timeout_minutes)
    if not ok:
        sys.exit(1)
    download_outputs()

    print("\n  Phase 4 (Kaggle PPO) complete. Models saved to:", MODELS_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout-minutes', type=int, default=90)
    parser.add_argument('--no-upload', action='store_true',
                        help='Skip dataset upload (reuse existing Kaggle dataset)')
    args = parser.parse_args()
    main(timeout_minutes=args.timeout_minutes, upload=not args.no_upload)
