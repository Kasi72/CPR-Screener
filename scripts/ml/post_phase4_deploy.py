"""
post_phase4_deploy.py — Watches for Phase 4 output files, then commits + deploys.

Runs in background after kaggle_phase4_runner.py completes.
Polls models/phase4_metrics.json until it appears (new mtime), then:
  1. git add model artifacts
  2. git commit
  3. vercel --prod

Usage:
    python scripts/ml/post_phase4_deploy.py
    python scripts/ml/post_phase4_deploy.py --timeout-minutes 180
"""

import os, sys, time, subprocess, json, argparse, pathlib

BASE       = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = BASE / 'models'
LOG_FILE   = BASE / 'scripts' / 'ml' / 'post_phase4_deploy.log'

TARGET_FILES = [
    'phase4_metrics.json',
    'ppo_policy_weights.json',
    'ppo_policy.zip',
]

COMMIT_FILES = [
    'models/phase4_metrics.json',
    'models/ppo_policy_weights.json',
    'models/ppo_policy.zip',
    'kaggle/phase4/cpr_phase4_kernel.py',
    'scripts/ml/kaggle_phase4_runner.py',
    'scripts/ml/post_phase4_deploy.py',
]


def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def run(args, cwd=BASE):
    log(f"  $ {' '.join(str(a) for a in args)}")
    r = subprocess.run(args, capture_output=True, text=True, cwd=str(cwd))
    if r.stdout.strip():
        log(f"  stdout: {r.stdout.strip()}")
    if r.stderr.strip():
        log(f"  stderr: {r.stderr.strip()}")
    return r


def wait_for_outputs(timeout_s):
    """Poll until all target files exist with recent mtime."""
    start     = time.time()
    deadline  = start + timeout_s
    poll_secs = 60
    attempt   = 0

    log(f"Watching for Phase 4 outputs in {MODELS_DIR} ...")
    log(f"  Targets: {TARGET_FILES}")
    log(f"  Timeout: {timeout_s/60:.0f} min")

    # Record baseline mtime so we can detect new files vs old ones
    baseline = {}
    for fn in TARGET_FILES:
        p = MODELS_DIR / fn
        baseline[fn] = p.stat().st_mtime if p.exists() else 0.0

    while time.time() < deadline:
        attempt += 1
        elapsed = (time.time() - start) / 60
        ready   = []
        missing = []

        for fn in TARGET_FILES:
            p = MODELS_DIR / fn
            if p.exists() and p.stat().st_mtime > baseline[fn]:
                ready.append(fn)
            else:
                missing.append(fn)

        log(f"[{elapsed:5.1f} min / attempt {attempt}]  ready={ready}  missing={missing}")

        if not missing:
            log("All Phase 4 output files detected!")
            return True

        time.sleep(poll_secs)

    log(f"Timeout after {timeout_s/60:.0f} min — Phase 4 outputs never appeared.")
    return False


def read_metrics():
    p = MODELS_DIR / 'phase4_metrics.json'
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def git_commit_and_deploy():
    log("=" * 60)
    log("Phase 4 complete — running git commit + Vercel deploy")
    log("=" * 60)

    # Read metrics for commit message
    m = read_metrics()
    sharpe_ppo   = m.get('sharpe_ppo',   'N/A')
    sharpe_full  = m.get('sharpe_full',  'N/A')
    sharpe_delta = m.get('sharpe_delta', 'N/A')
    avg_size     = m.get('avg_size',     'N/A')

    log(f"Metrics: sharpe_ppo={sharpe_ppo}, sharpe_full={sharpe_full}, "
        f"delta={sharpe_delta}, avg_size={avg_size}")

    # Stage files (only those that exist)
    staged = []
    for rel in COMMIT_FILES:
        p = BASE / rel
        if p.exists():
            r = run(['git', 'add', rel])
            if r.returncode == 0:
                staged.append(rel)
            else:
                log(f"  WARNING: git add failed for {rel}")
        else:
            log(f"  SKIP: {rel} not found")

    if not staged:
        log("ERROR: nothing staged — aborting commit.")
        return False

    # git status check
    run(['git', 'status', '--short'])

    commit_msg = (
        f"feat: Phase 4 PPO — Sharpe-delta reward, lgbm2c state, 200k timesteps\n\n"
        f"Results:\n"
        f"  sharpe_ppo:   {sharpe_ppo}\n"
        f"  sharpe_full:  {sharpe_full}\n"
        f"  sharpe_delta: {sharpe_delta}\n"
        f"  avg_size:     {avg_size}\n\n"
        f"Changes:\n"
        f"  - lgbm2c_score replaces regime_score in PPO state (65-feat ML confidence)\n"
        f"  - Sharpe-delta reward replaces P&L reward (rolling window 50)\n"
        f"  - 200k timesteps (up from 100k), early-stop 8 evals no improvement\n"
        f"  - 58-feature FEATURE_COLS matches Phase 2c exactly\n\n"
        f"Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    )

    r = run(['git', 'commit', '-m', commit_msg])
    if r.returncode != 0:
        log("ERROR: git commit failed.")
        return False

    log("Git commit successful.")

    # Push to remote
    log("Pushing to remote ...")
    r = run(['git', 'push'])
    if r.returncode != 0:
        log("WARNING: git push failed — commit is local only.")

    # Vercel deploy
    log("Deploying to Vercel (--prod) ...")
    r = run(['vercel', '--prod', '--yes'])
    if r.returncode == 0:
        log("Vercel deploy successful!")
        # Extract URL from output
        for line in (r.stdout + r.stderr).splitlines():
            if 'vercel.app' in line or 'Production' in line:
                log(f"  {line.strip()}")
    else:
        log("ERROR: Vercel deploy failed — check output above.")
        return False

    return True


def main(timeout_minutes=180):
    log("=" * 60)
    log("  post_phase4_deploy.py — Phase 4 Post-Completion Watcher")
    log("=" * 60)

    ok = wait_for_outputs(timeout_s=timeout_minutes * 60)
    if not ok:
        log("FAILED: timed out waiting for Phase 4 outputs.")
        sys.exit(1)

    # Small grace period so runner finishes writing all files
    log("Waiting 10s grace period ...")
    time.sleep(10)

    ok = git_commit_and_deploy()
    if not ok:
        log("FAILED: commit/deploy step failed.")
        sys.exit(1)

    log("Done. Phase 4 fully deployed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout-minutes', type=int, default=180,
                        help='Max wait for Phase 4 outputs (default 180 min)')
    args = parser.parse_args()
    main(timeout_minutes=args.timeout_minutes)
