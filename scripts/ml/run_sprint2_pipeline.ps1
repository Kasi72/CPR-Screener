# run_sprint2_pipeline.ps1
# Full Sprint 2 pipeline: feature augmentation → Phase 2c retrain → score → Phase 3
#
# Usage (from repo root):
#   .\scripts\ml\run_sprint2_pipeline.ps1             # full pipeline
#   .\scripts\ml\run_sprint2_pipeline.ps1 -SkipAugment  # if CSV already patched
#   .\scripts\ml\run_sprint2_pipeline.ps1 -StartFrom phase3  # skip 2c retrain

param(
    [switch]$SkipAugment,
    [ValidateSet('augment','phase2c','score','phase3')]
    [string]$StartFrom = 'augment'
)

$ErrorActionPreference = 'Stop'
$REPO = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$PYTHON = 'python'

function Banner([string]$msg) {
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
}

function RunStep([string]$label, [string]$script, [string[]]$args = @()) {
    Banner $label
    $cmd = @($PYTHON, $script) + $args
    Write-Host "CMD: $($cmd -join ' ')" -ForegroundColor DarkGray
    $t = [System.Diagnostics.Stopwatch]::StartNew()
    & $PYTHON $script @args
    $exit = $LASTEXITCODE
    $t.Stop()
    $elapsed = [Math]::Round($t.Elapsed.TotalMinutes, 1)
    if ($exit -ne 0) {
        Write-Host "FAILED (exit $exit) after ${elapsed}min" -ForegroundColor Red
        exit $exit
    }
    Write-Host "DONE in ${elapsed}min" -ForegroundColor Green
}

Set-Location $REPO

# ── Step 1: Augment signal_dataset.csv with Sprint 2 features ─────────────────
if ($StartFrom -eq 'augment' -and -not $SkipAugment) {
    RunStep "STEP 1/4 - Sprint 2 Feature Augmentation (10-20 min)" `
        "scripts\ml\add_sprint2_features.py"
} else {
    Write-Host "Skipping augmentation." -ForegroundColor Yellow
}

# ── Step 2: Phase 2c retrain on Kaggle (~90-120 min) ──────────────────────────
if ($StartFrom -in @('augment','phase2c')) {
    Banner "STEP 2/4 - Phase 2c Retrain on Kaggle (53 features)"
    Write-Host "  Upload + push kernel + poll + download outputs" -ForegroundColor DarkGray
    Write-Host "  Estimated time: 90-130 min" -ForegroundColor DarkGray
    & $PYTHON scripts\ml\kaggle_phase2c_runner.py
    if ($LASTEXITCODE -ne 0) { Write-Host "Phase 2c FAILED" -ForegroundColor Red; exit 1 }
    Write-Host "Phase 2c complete." -ForegroundColor Green

    # Show metrics
    $mpath = Join-Path $REPO "models\phase2c_metrics.json"
    if (Test-Path $mpath) {
        $m = Get-Content $mpath | ConvertFrom-Json
        Write-Host ""
        Write-Host "Phase 2c Metrics:" -ForegroundColor Cyan
        Write-Host "  global test AUC : $($m.test_auc)"
        Write-Host "  best CV AUC     : $($m.best_cv_auc)"
        Write-Host "  n_features      : $($m.n_features)"
        Write-Host "  n_rows          : $($m.n_rows)"
    }
}

# ── Step 3: Re-score signal_dataset.csv with new Phase 2c models ───────────────
if ($StartFrom -in @('augment','phase2c','score')) {
    RunStep "STEP 3/4 - Score signal_dataset.csv with new Phase 2c models (~5 min)" `
        "scripts\ml\score_p2c.py"
}

# ── Step 4: Phase 3 retrain on Kaggle (~60-90 min) ────────────────────────────
if ($StartFrom -in @('augment','phase2c','score','phase3')) {
    Banner "STEP 4/4 - Phase 3 Retrain on Kaggle (LSTM + stacking, 53 features)"
    Write-Host "  LSTM: 10 features, hidden_dim=128, epochs=80, cap=400K" -ForegroundColor DarkGray
    Write-Host "  Estimated time: 60-90 min" -ForegroundColor DarkGray
    & $PYTHON scripts\ml\launch_phase3_after_2c.py --timeout-minutes 180
    if ($LASTEXITCODE -ne 0) { Write-Host "Phase 3 FAILED" -ForegroundColor Red; exit 1 }

    # Show metrics
    $p3path = Join-Path $REPO "models\phase3_metrics.json"
    if (Test-Path $p3path) {
        $m = Get-Content $p3path | ConvertFrom-Json
        Write-Host ""
        Write-Host "Phase 3 Metrics:" -ForegroundColor Cyan
        Write-Host "  auc_p2b   : $($m.auc_p2b)"
        Write-Host "  auc_p2c   : $($m.auc_p2c)"
        Write-Host "  auc_lstm  : $($m.auc_lstm)"
        Write-Host "  auc_stack : $($m.auc_stack)"
        Write-Host "  threshold : $($m.threshold) (prec $($m.threshold_precision))"
    }
}

Banner "PIPELINE COMPLETE"
Write-Host "Next: deploy to Vercel with:  vercel deploy --prod --yes" -ForegroundColor Yellow

