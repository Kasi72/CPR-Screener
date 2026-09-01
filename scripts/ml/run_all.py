"""
run_all.py — Run all training phases in sequence.

Usage:
    python scripts/ml/run_all.py [--skip-dataset] [--local-phase3]

Steps:
    0. build_dataset.py          (re-run backtest to generate signal_dataset.csv)
    1. train_phase1.py           (HMM + LightGBM)
    2. train_phase2.py           (SHAP gate weights + Conformal calibration)
   2b. kaggle_phase2b_runner.py  (LightGBM HPO signal scorer — runs on Kaggle CPU)
    3. kaggle_phase3_runner.py   (LSTM + Stacking — runs on Kaggle T4 GPU)
       OR train_phase3.py        (local CPU fallback with --local-phase3)
    4. kaggle_phase4_runner.py   (PPO position sizing — runs on Kaggle)
       OR train_phase4.py        (local CPU fallback with --local-phase4)

Phase 2b/3/4 Kaggle flags:
    --skip-phase2b               Skip Phase 2b (LightGBM HPO — runs on Kaggle)
    --local-phase3               Force local CPU training for Phase 3
    --local-phase4               Force local CPU training for Phase 4
    --skip-upload                Re-use existing Kaggle dataset for Phase 3
    --timeout-minutes N          Kaggle poll timeout for Phase 3 (default: 120)
    --p4-timeout-minutes N       Kaggle poll timeout for Phase 4 (default: 60)
    --p2b-timeout-minutes N      Kaggle poll timeout for Phase 2b (default: 90)
"""

import subprocess, sys, os, time, shutil

BASE    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(BASE, 'scripts', 'ml')


def _kaggle_available():
    """Return True if kaggle CLI is on PATH and credentials are configured."""
    try:
        r = subprocess.run(['kaggle', '--version'], capture_output=True, text=True)
        if r.returncode != 0:
            return False
        cred = os.path.join(os.path.expanduser('~'), '.kaggle', 'kaggle.json')
        return os.path.exists(cred)
    except FileNotFoundError:
        return False


def run(script, label, extra_args=None):
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"{'='*60}\n")
    t0  = time.time()
    cmd = [sys.executable, os.path.join(SCRIPTS, script)]
    if extra_args:
        cmd.extend(extra_args)
    ret = subprocess.run(cmd, cwd=BASE, check=False)
    elapsed = time.time() - t0
    ok = ret.returncode == 0
    status = '✓' if ok else f'✗ (exit {ret.returncode})'
    print(f"\n  {status}  {label}  [{elapsed/60:.1f} min]")
    return ok, elapsed


def main():
    args          = sys.argv[1:]
    skip_dataset  = '--skip-dataset'    in args
    skip_p1       = '--skip-phase1'     in args
    skip_p2       = '--skip-phase2'     in args
    skip_p2b      = '--skip-phase2b'    in args
    skip_p3       = '--skip-phase3'     in args
    skip_p4       = '--skip-phase4'     in args
    local_p3      = '--local-phase3'    in args
    local_p4      = '--local-phase4'    in args
    skip_upload   = '--skip-upload'     in args

    timeout_min     = 120
    p4_timeout_min  = 60
    p2b_timeout_min = 90
    for a in args:
        if a.startswith('--timeout-minutes='):
            timeout_min = int(a.split('=')[1])
        if a.startswith('--p4-timeout-minutes='):
            p4_timeout_min = int(a.split('=')[1])
        if a.startswith('--p2b-timeout-minutes='):
            p2b_timeout_min = int(a.split('=')[1])

    # Decide execution modes
    kaggle_ok     = _kaggle_available()
    use_kaggle_p2b = kaggle_ok and not skip_p2b
    use_kaggle_p3  = (not local_p3) and kaggle_ok
    use_kaggle_p4  = (not local_p4) and kaggle_ok
    if not skip_p2b:
        if use_kaggle_p2b:
            print('\n  Phase 2b → Kaggle CPU  (use --skip-phase2b to skip)')
        else:
            print('\n  Phase 2b → SKIPPED  (kaggle CLI not available)')
    if not skip_p3:
        if use_kaggle_p3:
            print("\n  Phase 3 → Kaggle GPU  (use --local-phase3 to run locally)")
        else:
            reason = "--local-phase3 flag" if local_p3 else "kaggle CLI not available"
            print(f"\n  Phase 3 → Local CPU  ({reason})")
    if not skip_p4:
        if use_kaggle_p4:
            print("\n  Phase 4 → Kaggle     (use --local-phase4 to run locally)")
        else:
            reason = "--local-phase4 flag" if local_p4 else "kaggle CLI not available"
            print(f"\n  Phase 4 → Local CPU  ({reason})")

    steps = []
    if not skip_dataset:
        steps.append(('build_dataset.py', 'Step 0: Build Dataset', None))
    if not skip_p1:
        steps.append(('train_phase1.py', 'Phase 1: HMM + LightGBM', None))
    if not skip_p2:
        steps.append(('train_phase2.py', 'Phase 2: SHAP + Conformal', None))
    if use_kaggle_p2b:
        p2b_extra = [f'--timeout-minutes={p2b_timeout_min}']
        steps.append(('kaggle_phase2b_runner.py',
                      'Phase 2b: LightGBM HPO Scorer (Kaggle CPU)', p2b_extra))
    if not skip_p3:
        if use_kaggle_p3:
            p3_extra = [f'--timeout-minutes={timeout_min}']
            if skip_upload:
                p3_extra.append('--skip-upload')
            steps.append(('kaggle_phase3_runner.py',
                          'Phase 3: LSTM + Stacking (Kaggle GPU)', p3_extra))
        else:
            steps.append(('train_phase3.py', 'Phase 3: LSTM + Stacking (local)', None))
    if not skip_p4:
        if use_kaggle_p4:
            p4_extra = [f'--timeout-minutes={p4_timeout_min}']
            steps.append(('kaggle_phase4_runner.py',
                          'Phase 4: PPO Sizer (Kaggle)', p4_extra))
        else:
            steps.append(('train_phase4.py', 'Phase 4: PPO Sizer (local)', None))

    print("\n" + "="*60)
    print("  Dr KKR CPR Screener — Full ML Training Pipeline")
    print("="*60)

    t_start  = time.time()
    results  = []
    failed_at = None
    for script, label, extra in steps:
        ok, elapsed = run(script, label, extra_args=extra)
        results.append((label, ok, elapsed))
        if not ok:
            failed_at = label
            break

    # Summary table
    total = time.time() - t_start
    print("\n" + "="*60)
    print("  PIPELINE SUMMARY")
    print("="*60)
    for label, ok, elapsed in results:
        mark = '✓' if ok else '✗'
        print(f"  {mark}  {label:<35} {elapsed/60:5.1f} min")
    print(f"  {'─'*50}")
    print(f"  Total: {total/60:.1f} min")
    print("="*60)

    if failed_at:
        print(f"\n  FAILED at: {failed_at}")
        print("  To resume:  python scripts/ml/run_all.py --skip-dataset --skip-phase1 ...")
        sys.exit(1)

    _restart_predict_server()


def _restart_predict_server():
    """Kill any running predict_server.py and start a fresh instance."""
    import platform
    server_script = os.path.join(BASE, 'predict_server.py')
    if not os.path.exists(server_script):
        print("  predict_server.py not found — skip restart.")
        return

    print("\n  Restarting predict_server.py…")
    if platform.system() == 'Windows':
        # Kill by port 5001 (most reliable — works even without WMIC/window title)
        net = subprocess.run(
            ['netstat', '-ano', '-p', 'tcp'],
            capture_output=True, text=True, check=False
        )
        for line in net.stdout.splitlines():
            if ':5001 ' in line and ('LISTENING' in line or 'ESTABLISHED' in line):
                pid = line.split()[-1].strip()
                if pid.isdigit() and int(pid) > 0:
                    subprocess.run(['taskkill', '/F', '/PID', pid],
                                   capture_output=True, check=False)
                    print(f"    Killed PID {pid} (port 5001)")
        # Fallback: kill by commandline match via wmic
        result = subprocess.run(
            ['wmic', 'process', 'where',
             'commandline like "%predict_server%"', 'get', 'processid'],
            capture_output=True, text=True, check=False
        )
        for line in result.stdout.splitlines():
            pid = line.strip()
            if pid.isdigit():
                subprocess.run(['taskkill', '/F', '/PID', pid],
                               capture_output=True, check=False)
                print(f"    Killed PID {pid} (commandline match)")
    else:
        subprocess.run(
            ['pkill', '-f', 'predict_server.py'],
            capture_output=True, check=False
        )

    time.sleep(2)
    log_path = os.path.join(BASE, 'predict_server.log')
    with open(log_path, 'a') as log:
        subprocess.Popen(
            [sys.executable, server_script],
            cwd=BASE,
            stdout=log, stderr=log,
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                           if platform.system() == 'Windows' else 0),
        )
    print(f"  predict_server.py started. Log → {log_path}")


if __name__ == '__main__':
    main()
