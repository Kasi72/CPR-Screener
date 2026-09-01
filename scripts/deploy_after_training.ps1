#!/usr/bin/env pwsh
# deploy_after_training.ps1
# Run after Kaggle training completes to commit models + deploy to Vercel.
# Usage: pwsh scripts/deploy_after_training.ps1

$ErrorActionPreference = 'Stop'
$ROOT = Split-Path -Parent $PSScriptRoot

Set-Location $ROOT

# ── 1. Verify expected model files ────────────────────────────────────────────
$required = @(
    'models/lgbm_scorer.txt',
    'models/shap_weights.json',
    'models/phase2b_metrics.json'
)
# Include Phase 4 models unless watcher signalled to skip (local poller timed out + Kaggle dl also failed)
if ($env:SKIP_P4_CHECK -ne '1') {
    $required += 'models/ppo_policy_weights.json'
    $required += 'models/phase4_metrics.json'
}

$missing = @()
foreach ($f in $required) {
    $path = Join-Path $ROOT $f
    if (-not (Test-Path $path) -or (Get-Item $path).Length -eq 0) {
        $missing += $f
    }
}

if ($missing.Count -gt 0) {
    Write-Host "MISSING or empty model files:"
    $missing | ForEach-Object { Write-Host "  $_" }
    Write-Host "Aborting — re-run after training completes."
    exit 1
}

Write-Host "All model files present. Proceeding."

# ── 2. Show Phase metrics ──────────────────────────────────────────────────────
$p2b = Get-Content (Join-Path $ROOT 'models/phase2b_metrics.json') | ConvertFrom-Json
Write-Host "Phase2b: CV AUC=$($p2b.best_cv_auc)  Val AUC=$($p2b.final_val_auc)  Runtime=$($p2b.runtime_min) min"

$p4content = Get-Content (Join-Path $ROOT 'models/phase4_metrics.json')
if ($p4content) { Write-Host "Phase4 metrics: $p4content" }

# ── 3. Git add + commit + push ────────────────────────────────────────────────
$date = Get-Date -Format 'yyyy-MM-dd'

# Stage code changes (Phase 2b runner, kernel, predict_server)
git add kaggle/phase2b/cpr_phase2b_kernel.py
git add kaggle/phase2b/kernel-metadata.json
git add scripts/ml/kaggle_phase2b_runner.py
git add predict_server.py
git add scripts/deploy_after_training.ps1

# Stage tracked model files
git add models/ppo_policy_weights.json
git add models/phase4_metrics.json
git add models/lgbm_scorer.txt
git add models/shap_weights.json
git add models/phase2b_metrics.json

$status = git status --short
if (-not $status) {
    Write-Host "Nothing to commit — already up to date."
} else {
    $msg = "feat: Phase2b LightGBM HPO scorer + Phase4 PPO models ($date)"
    git commit -m $msg
    Write-Host "Committed."
    git push origin master
    Write-Host "Pushed to origin/master."
}

# ── 4. Vercel deploy ──────────────────────────────────────────────────────────
Write-Host "Deploying to Vercel..."
vercel --prod --yes
Write-Host "Vercel deploy complete."
