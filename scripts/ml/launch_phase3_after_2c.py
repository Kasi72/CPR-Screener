"""
launch_phase3_after_2c.py
Polls until Phase 2c v2 models are present, then auto-launches Phase 3.

Usage:
    python scripts/ml/launch_phase3_after_2c.py
    python scripts/ml/launch_phase3_after_2c.py --skip-upload   # re-use existing Kaggle dataset
"""

import os, sys, time, argparse

BASE       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE, 'models')

# Phase 2c v2 is done when ALL regime models + global are present
PHASE2C_REQUIRED = [
    'lgbm2c_global.txt',
    'lgbm2c_regime_0.txt',
    'lgbm2c_regime_1.txt',
    'lgbm2c_regime_2.txt',
    'lgbm2c_regime_3.txt',
    'phase2c_metrics.json',
    'shap_weights2c.json',
]

POLL_INTERVAL = 60   # seconds


def all_present():
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in PHASE2C_REQUIRED)


def newest_mtime():
    times = [os.path.getmtime(os.path.join(MODELS_DIR, f))
             for f in PHASE2C_REQUIRED
             if os.path.exists(os.path.join(MODELS_DIR, f))]
    return max(times) if times else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-upload', action='store_true',
                        help='Skip Kaggle dataset upload (re-use existing version)')
    parser.add_argument('--timeout-minutes', type=int, default=180,
                        help='Phase 3 poll timeout in minutes (default: 180)')
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase 3 Auto-Launcher")
    print("  Waiting for Phase 2c v2 models to appear in models/ ...")
    print("=" * 60)

    # If already present, check if they're fresh (written in last 4 hours)
    if all_present():
        age_min = (time.time() - newest_mtime()) / 60
        print(f"  Phase 2c models already present (newest {age_min:.0f} min old).")
        print("  Proceeding to Phase 3 immediately.")
    else:
        waited = 0
        while not all_present():
            missing = [f for f in PHASE2C_REQUIRED
                       if not os.path.exists(os.path.join(MODELS_DIR, f))]
            print(f"  [{waited // 60:.0f} min] Waiting... missing: {missing}")
            time.sleep(POLL_INTERVAL)
            waited += POLL_INTERVAL
            if waited > 7200:   # 2h hard timeout
                print("  ERROR: Phase 2c did not complete within 2 hours. Aborting.")
                sys.exit(1)
        print(f"\n  Phase 2c models detected after {waited // 60:.0f} min. Launching Phase 3...")

    # Import and run Phase 3 runner
    sys.path.insert(0, BASE)
    from scripts.ml.kaggle_phase3_runner import (
        package_inputs, upload_dataset, wait_for_dataset_ready,
        push_kernel, poll_kernel, download_outputs
    )
    import tempfile

    staging = tempfile.mkdtemp(prefix='kaggle_p3_stage_')
    try:
        package_inputs(staging, skip_ohlcv=False)
        if not args.skip_upload:
            upload_dataset(staging)
            wait_for_dataset_ready()
        else:
            print("  [2/5] Skipping dataset upload (--skip-upload).")
        push_kernel()
        ok = poll_kernel(timeout_minutes=args.timeout_minutes)
        if ok:
            download_outputs()
            print("\n  Phase 3 complete!")
        else:
            print("\n  Phase 3 kernel failed or timed out.")
            sys.exit(1)
    finally:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == '__main__':
    main()
