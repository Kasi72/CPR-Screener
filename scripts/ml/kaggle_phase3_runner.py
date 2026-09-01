"""
kaggle_phase3_runner.py — Orchestrates Phase 3 LSTM training on Kaggle GPU.

Called by run_all.py as a drop-in replacement for running train_phase3.py locally.

Flow:
  1. Package inputs  → staging dir (signal_dataset.csv + Phase 1/2 models + OHLCV)
  2. Upload inputs   → Kaggle dataset  drkasi/cpr-screener-phase3-inputs
  3. Push kernel     → Kaggle kernel   drkasi/cpr-phase3-lstm
  4. Poll status     → wait for 'complete' (2-hour timeout)
  5. Download output → /kaggle/working/ files copied to local models/

Usage (standalone):
    python scripts/ml/kaggle_phase3_runner.py

Flags:
    --timeout-minutes N  Override 2h poll timeout (default: 120)
    --skip-upload        Re-use existing Kaggle dataset version (faster re-runs)
"""

import os, sys, json, time, shutil, subprocess, tempfile, glob, zipfile, argparse

BASE       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KAGGLE_DIR = os.path.join(BASE, 'kaggle')
MODELS_DIR = os.path.join(BASE, 'models')

DATASET_SLUG = 'drkasi/cpr-screener-phase3-inputs'
KERNEL_SLUG  = 'drkasi/cpr-phase3-lstm'

# Max size (bytes) for OHLCV optional upload — skip if larger
OHLCV_MAX_BYTES = 2 * 1024 ** 3   # 2 GB

# Paths
OHLCV_PATH     = r"C:\Users\drkkr\Downloads\NIFTY 500 OHLCV\ALL_SYMBOLS_OHLCV.csv"
SIGNAL_CSV     = os.path.join(MODELS_DIR, 'signal_dataset.csv')


# ── Subprocess helper ─────────────────────────────────────────────────────────

def _run(args, check=True, capture=False):
    r = subprocess.run(args, capture_output=capture, text=True)
    if check and r.returncode != 0:
        msg = (r.stderr or r.stdout or '').strip()
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(args)}\n{msg}")
    return r


def _kaggle(*args, capture=False):
    return _run(['kaggle'] + list(args), capture=capture)


# ── Step 1: Package inputs ────────────────────────────────────────────────────

def package_inputs(staging_dir, skip_ohlcv=False):
    """Copy all Phase 3 inputs into staging_dir for Kaggle dataset upload."""
    print("  [1/5] Packaging inputs ...")

    # Validate required files exist
    if not os.path.exists(SIGNAL_CSV):
        raise FileNotFoundError(
            f"signal_dataset.csv not found at {SIGNAL_CSV}.\n"
            "Run Step 0 (build_dataset.py) first."
        )
    lgbm_path = os.path.join(MODELS_DIR, 'lgbm_model.txt')
    if not os.path.exists(lgbm_path):
        raise FileNotFoundError(
            f"lgbm_model.txt not found at {lgbm_path}.\n"
            "Run Phase 1 (train_phase1.py) first."
        )

    # dataset-metadata.json (required by kaggle datasets create/version)
    meta = {
        "title": "CPR Screener Phase 3 Inputs",
        "id":    DATASET_SLUG,
        "licenses": [{"name": "other"}]
    }
    with open(os.path.join(staging_dir, 'dataset-metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # signal_dataset.csv → staging root
    size_mb = os.path.getsize(SIGNAL_CSV) / 1024 ** 2
    print(f"    Copying signal_dataset.csv ({size_mb:.0f} MB) ...")
    shutil.copy2(SIGNAL_CSV, os.path.join(staging_dir, 'signal_dataset.csv'))

    # models/ directory (Phase 1 + 2 artifacts)
    models_out = os.path.join(staging_dir, 'models')
    os.makedirs(models_out, exist_ok=True)

    model_files = (
        ['lgbm_model.txt', 'phase1_metrics.json', 'phase2_metrics.json',
         'hmm_params.json', 'hmm_posteriors.json', 'soft_blend_config.json',
         'conformal_calibration.json', 'gate_weights.json']
        + [f'lgbm_regime_{s}.txt' for s in range(4)]
        + [f'lgbm_rule{i}.txt'    for i in range(1, 12)]
    )
    copied = []
    for fn in model_files:
        src = os.path.join(MODELS_DIR, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(models_out, fn))
            copied.append(fn)
    print(f"    Models copied: {len(copied)} files")

    # OHLCV data (optional — enables real 30-bar sequences, better LSTM)
    if not skip_ohlcv and os.path.exists(OHLCV_PATH):
        ohlcv_size = os.path.getsize(OHLCV_PATH)
        if ohlcv_size <= OHLCV_MAX_BYTES:
            ohlcv_out = os.path.join(staging_dir, 'ohlcv')
            os.makedirs(ohlcv_out, exist_ok=True)
            ohlcv_mb = ohlcv_size / 1024 ** 2
            print(f"    Copying OHLCV data ({ohlcv_mb:.0f} MB) ...")
            shutil.copy2(OHLCV_PATH, os.path.join(ohlcv_out, 'ALL_SYMBOLS_OHLCV.csv'))
        else:
            print(f"    OHLCV ({ohlcv_size/1024**3:.1f} GB) exceeds 2 GB limit — skipped.")
    else:
        if not os.path.exists(OHLCV_PATH):
            print("    OHLCV not found — kernel will use tabular proxy sequences.")

    print("    Packaging done.")


# ── Step 2: Upload dataset ────────────────────────────────────────────────────

def upload_dataset(staging_dir):
    """Create or add a new version of the Kaggle dataset."""
    print("  [2/5] Uploading to Kaggle dataset ...")
    ts = time.strftime('%Y-%m-%d %H:%M')

    # Check if dataset already exists by listing it
    r = _kaggle('datasets', 'list', '--search', 'cpr-screener-phase3-inputs',
                '--user', 'drkasi', capture=True)
    exists = 'cpr-screener-phase3-inputs' in (r.stdout or '')

    if exists:
        print(f"    Dataset exists — uploading new version ...")
        _kaggle('datasets', 'version', '-p', staging_dir,
                '-m', f'Phase3 inputs {ts}', '--dir-mode', 'zip')
    else:
        print("    Creating new dataset ...")
        _kaggle('datasets', 'create', '-p', staging_dir, '--dir-mode', 'zip')

    print("    Dataset upload complete.")


# ── Step 3: Push kernel ───────────────────────────────────────────────────────

def push_kernel():
    """Push the kernel from the kaggle/ directory."""
    print("  [3/5] Pushing kernel to Kaggle ...")
    _kaggle('kernels', 'push', '-p', KAGGLE_DIR)
    print("    Kernel pushed. Training started on Kaggle GPU.")


# ── Step 4: Poll for completion ───────────────────────────────────────────────

def poll_kernel(timeout_minutes=120):
    """Poll kernel status every 60 seconds until complete or error."""
    print(f"  [4/5] Polling kernel status (timeout: {timeout_minutes} min) ...")
    deadline = time.time() + timeout_minutes * 60
    interval = 60   # seconds between polls
    start    = time.time()

    while time.time() < deadline:
        r = _kaggle('kernels', 'status', KERNEL_SLUG, capture=True)
        output = (r.stdout or '').strip()
        # Status line looks like: "drkasi/cpr-phase3-lstm ... running"
        status = 'unknown'
        for token in ['complete', 'running', 'queued', 'error', 'cancelAcknowledged']:
            if token in output.lower():
                status = token
                break

        elapsed = (time.time() - start) / 60
        print(f"    [{elapsed:5.1f} min]  status: {status}", flush=True)

        if status == 'complete':
            print("    Kernel complete!")
            return True
        if status in ('error', 'cancelacknowledged'):
            print(f"    Kernel failed with status: {status}")
            print(f"    Output:\n{output}")
            return False

        time.sleep(interval)

    print(f"    Timeout after {timeout_minutes} minutes.")
    return False


# ── Step 5: Download outputs ──────────────────────────────────────────────────

def download_outputs():
    """Download kernel output zip and copy model files to local models/."""
    print("  [5/5] Downloading kernel outputs ...")
    out_dir = tempfile.mkdtemp(prefix='kaggle_p3_out_')
    try:
        _kaggle('kernels', 'output', KERNEL_SLUG, '-p', out_dir)

        # kaggle downloads a zip file named after the kernel slug
        zips = glob.glob(os.path.join(out_dir, '*.zip'))
        if not zips:
            # Sometimes outputs are already extracted
            files = glob.glob(os.path.join(out_dir, '*'))
            print(f"    No zip found; files in out_dir: {files}")
        else:
            zip_path = zips[0]
            print(f"    Extracting {os.path.basename(zip_path)} ...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(out_dir)

        # Copy model outputs to local MODELS_DIR
        target_files = [
            'lstm_model.pt',
            'stacking_weights.json',
            'meta_lgbm.txt',
            'phase3_metrics.json',
        ]
        copied = []
        for fn in target_files:
            # Search recursively in case zip had sub-dirs
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

        print(f"    All {len(copied)} output files copied to {MODELS_DIR}")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(timeout_minutes=120, skip_upload=False):
    print("\n" + "=" * 60)
    print("  Phase 3: Kaggle GPU Runner")
    print("=" * 60)

    staging_dir = tempfile.mkdtemp(prefix='kaggle_p3_stage_')
    try:
        package_inputs(staging_dir, skip_ohlcv=skip_upload)

        if not skip_upload:
            upload_dataset(staging_dir)
        else:
            print("  [2/5] Skipping dataset upload (--skip-upload).")

        push_kernel()
        ok = poll_kernel(timeout_minutes=timeout_minutes)
        if not ok:
            sys.exit(1)
        download_outputs()
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    print("\n  Phase 3 (Kaggle GPU) complete. Models saved to:", MODELS_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout-minutes', type=int, default=120)
    parser.add_argument('--skip-upload', action='store_true')
    args = parser.parse_args()
    main(timeout_minutes=args.timeout_minutes, skip_upload=args.skip_upload)
