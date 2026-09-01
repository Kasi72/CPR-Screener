# auto_retrain_cpr.ps1 — Monthly CPR Screener full retrain.
#
# Phase 3 (LSTM + Stacking) auto-routes to Kaggle T4 GPU.
# Reliable: lock file, log rotation, working-dir pinned, full error trapping.

$PROJECT  = "D:\Claude code\nse-screener"
$PYTHON   = "C:\Users\drkkr\AppData\Local\Programs\Python\Python310\python.exe"
$RUNALL   = "$PROJECT\scripts\ml\run_all.py"
$LOG      = "$PROJECT\auto_retrain_cpr.log"
$LOCK     = "$PROJECT\auto_retrain_cpr.lock"
$STAMP    = "$PROJECT\auto_retrain_cpr.last_run"   # records last successful run date

$ErrorActionPreference = "Stop"

# ── Lock: prevent concurrent runs ─────────────────────────────────────────────
if (Test-Path $LOCK) {
    $age = (Get-Date) - (Get-Item $LOCK).LastWriteTime
    if ($age.TotalHours -lt 6) {
        Write-Host "Lock file exists (age $([int]$age.TotalMinutes) min) — another run in progress. Exiting."
        exit 0
    }
    Write-Host "Stale lock (age $([int]$age.TotalHours)h) — removing and continuing."
    Remove-Item $LOCK -Force
}
"$((Get-Date).ToString('o'))" | Set-Content $LOCK

# ── Log rotation: keep last 500 lines ─────────────────────────────────────────
if (Test-Path $LOG) {
    $lines = Get-Content $LOG -ErrorAction SilentlyContinue
    if ($lines.Count -gt 500) {
        $lines[-400..-1] | Set-Content $LOG
    }
}

function Write-Log {
    param([string]$Msg)
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Msg"
    Write-Host $line
    Add-Content -Path $LOG -Value $line
}

# ── Ensure correct working directory ──────────────────────────────────────────
Set-Location $PROJECT

try {
    Write-Log "========================================================"
    Write-Log "CPR Monthly Retrain — START"
    Write-Log "Python: $PYTHON"
    Write-Log "========================================================"

    # Verify prerequisites
    if (-not (Test-Path $PYTHON)) {
        throw "Python not found at $PYTHON"
    }
    if (-not (Test-Path $RUNALL)) {
        throw "run_all.py not found at $RUNALL"
    }

    # ── Run pipeline ──────────────────────────────────────────────────────────
    # Phase 3 auto-routes to Kaggle GPU (requires Kaggle CLI + kaggle.json).
    # Remove --skip-dataset only after downloading fresh OHLCV data.
    Write-Log "Running: python scripts/ml/run_all.py --skip-dataset"

    $proc = Start-Process -FilePath $PYTHON `
        -ArgumentList "$RUNALL --skip-dataset" `
        -WorkingDirectory $PROJECT `
        -NoNewWindow -PassThru -Wait `
        -RedirectStandardOutput "$env:TEMP\cpr_retrain_stdout.txt" `
        -RedirectStandardError  "$env:TEMP\cpr_retrain_stderr.txt"

    # Stream captured output to log
    if (Test-Path "$env:TEMP\cpr_retrain_stdout.txt") {
        Get-Content "$env:TEMP\cpr_retrain_stdout.txt" | ForEach-Object { Add-Content $LOG $_ }
    }
    if (Test-Path "$env:TEMP\cpr_retrain_stderr.txt") {
        $errLines = Get-Content "$env:TEMP\cpr_retrain_stderr.txt" -ErrorAction SilentlyContinue
        if ($errLines) { $errLines | ForEach-Object { Add-Content $LOG "STDERR: $_" } }
    }

    if ($proc.ExitCode -ne 0) {
        throw "Pipeline exited with code $($proc.ExitCode)"
    }

    Write-Log "Pipeline COMPLETE"

    # ── Commit + push updated models ──────────────────────────────────────────
    Write-Log "Staging model artifacts ..."
    $modelPatterns = @(
        "models/lgbm_model.txt",   "models/lgbm_regime_*.txt", "models/lgbm_rule*.txt",
        "models/meta_lgbm.txt",    "models/stacking_weights.json",
        "models/phase1_metrics.json", "models/phase2_metrics.json",
        "models/phase3_metrics.json", "models/phase4_metrics.json",
        "models/hmm_params.json",  "models/hmm_posteriors.json",
        "models/gate_weights.json","models/conformal_calibration.json",
        "models/soft_blend_config.json"
    )
    foreach ($pat in $modelPatterns) {
        $files = Resolve-Path $pat -ErrorAction SilentlyContinue
        if ($files) { git -C $PROJECT add $files 2>&1 | Add-Content $LOG }
    }

    $staged = git -C $PROJECT diff --cached --name-only 2>$null
    if ($staged) {
        $tag = Get-Date -Format 'yyyy-MM-dd'
        git -C $PROJECT commit -m "chore: monthly model retrain $tag [auto]" 2>&1 | Add-Content $LOG
        git -C $PROJECT push 2>&1 | Add-Content $LOG
        Write-Log "Models committed and pushed — GitHub auto-deploys to Vercel."
    } else {
        Write-Log "No model changes staged — nothing to commit."
    }

    # Record successful run timestamp
    (Get-Date).ToString('yyyy-MM-dd HH:mm:ss') | Set-Content $STAMP
    Write-Log "CPR Monthly Retrain DONE."
    exit 0

} catch {
    Write-Log "ERROR: $_"
    Write-Log "CPR Monthly Retrain FAILED."
    exit 1
} finally {
    # Always release lock
    Remove-Item $LOCK -Force -ErrorAction SilentlyContinue
    Remove-Item "$env:TEMP\cpr_retrain_stdout.txt" -Force -ErrorAction SilentlyContinue
    Remove-Item "$env:TEMP\cpr_retrain_stderr.txt" -Force -ErrorAction SilentlyContinue
}
