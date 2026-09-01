#!/usr/bin/env pwsh
# watch_and_deploy.ps1
# Watches Phase 4 + Phase 2b poll logs; auto-deploys when both finish.
# Usage: pwsh scripts/watch_and_deploy.ps1

$P4LOG  = "$env:TEMP\phase4_poll3.log"
$P2BLOG = "$env:TEMP\phase2b_poll2.log"
$ROOT   = Split-Path -Parent $PSScriptRoot
$DEPLOY = Join-Path $ROOT 'scripts\deploy_after_training.ps1'

Write-Host "Watching logs..."
Write-Host "  Phase 4 : $P4LOG"
Write-Host "  Phase 2b: $P2BLOG"

$interval = 30
$p4done   = $false
$p2bdone  = $false
$p4error  = $false
$p2berror = $false

while (-not ($p4done -and $p2bdone) -and -not ($p4error -or $p2berror)) {
    Start-Sleep -Seconds $interval

    if (-not $p4done) {
        if (Test-Path $P4LOG) {
            $tail = Get-Content $P4LOG -Tail 5
            if ($tail -match 'DONE:phase4') { $p4done = $true; Write-Host "[$(Get-Date -f HH:mm:ss)] Phase 4 DONE" }
            # Match only the terminal error lines (not the startup banner "(timeout N min)")
            if ($tail -match '^Timeout\.$|^    Timeout after|FAILED:|Kernel failed:') { $p4error = $true; Write-Host "[$(Get-Date -f HH:mm:ss)] Phase 4 ERROR/TIMEOUT"; $tail }
        }
    }
    if (-not $p2bdone) {
        if (Test-Path $P2BLOG) {
            $tail = Get-Content $P2BLOG -Tail 5
            if ($tail -match 'Phase 2b download done') { $p2bdone = $true; Write-Host "[$(Get-Date -f HH:mm:ss)] Phase 2b DONE" }
            if ($tail -match '^Timeout\.$|^    Timeout after|FAILED:|Kernel failed:') { $p2berror = $true; Write-Host "[$(Get-Date -f HH:mm:ss)] Phase 2b ERROR/TIMEOUT"; $tail }
        }
    }

    $now = Get-Date -Format 'HH:mm:ss'
    $p4s  = if ($p4done) { 'done' } elseif ($p4error) { 'ERROR' } else { 'waiting' }
    $p2bs = if ($p2bdone) { 'done' } elseif ($p2berror) { 'ERROR' } else { 'waiting' }
    Write-Host "[$now] P4=$p4s  P2b=$p2bs"
}

if ($p2berror) {
    Write-Host "Phase 2b kernel failed. Manual intervention needed."
    exit 1
}

# Phase 4 local poller may timeout while Kaggle kernel still runs.
# If P4 timed out locally, attempt direct download then deploy.
if ($p4error) {
    Write-Host "[$(Get-Date -f HH:mm:ss)] Phase 4 local poller timed out — attempting direct download from Kaggle..."
    $py = "C:\Users\drkkr\AppData\Local\Programs\Python\Python310\python.exe"
    $dlScript = @'
import subprocess, glob, shutil, zipfile, tempfile, os, json, sys, time

KERNEL_SLUG = 'drkasi/cpr-phase-4-ppo-position-sizing'
MODELS_DIR  = r'D:\Claude code\nse-screener\models'

# Wait up to 4 extra hours for kernel to complete before giving up
deadline = time.time() + 4 * 3600
while time.time() < deadline:
    r = subprocess.run(['kaggle', 'kernels', 'status', KERNEL_SLUG], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().lower()
    if 'complete' in out: break
    if 'error' in out or 'cancel' in out:
        print('Phase 4 kernel errored on Kaggle:', out); sys.exit(1)
    elapsed = (deadline - time.time()) / 60
    print(f'Waiting for Phase 4 to complete... ({elapsed:.0f} min budget left)', flush=True)
    time.sleep(120)
else:
    print('Gave up waiting for Phase 4.'); sys.exit(1)

out_dir = tempfile.mkdtemp(prefix='p4dl_')
subprocess.run(['kaggle', 'kernels', 'output', KERNEL_SLUG, '-p', out_dir])
zips = glob.glob(os.path.join(out_dir, '*.zip'))
if zips:
    with zipfile.ZipFile(zips[0]) as zf: zf.extractall(out_dir)
for fn in ['ppo_policy_weights.json', 'phase4_metrics.json', 'ppo_policy.zip']:
    hits = glob.glob(os.path.join(out_dir, '**', fn), recursive=True)
    if hits:
        shutil.copy2(hits[0], os.path.join(MODELS_DIR, fn))
        print(f'OK {fn}')
    else:
        print(f'MISS {fn}')
shutil.rmtree(out_dir, ignore_errors=True)
print('DONE:phase4')
'@
    $tmpPy = "$env:TEMP\p4_wait_dl.py"
    $dlScript | Set-Content $tmpPy -Encoding UTF8
    & $py $tmpPy
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Phase 4 download also failed. Deploy Phase 2b models only (P4 excluded)."
        # Deploy without P4 models — remove P4 from required list in deploy script context
        $env:SKIP_P4_CHECK = '1'
    }
}

Write-Host "Running deploy..."
& pwsh $DEPLOY
