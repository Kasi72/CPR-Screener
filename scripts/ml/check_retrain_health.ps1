# check_retrain_health.ps1
# Usage: powershell -File "D:\Claude code\nse-screener\scripts\ml\check_retrain_health.ps1"

$ProjectDir = "D:\Claude code\nse-screener"
$LogFile    = "$ProjectDir\scripts\ml\run_all_retrain.log"
$LockFile   = "$ProjectDir\scripts\ml\retrain.lock"
$ModelsDir  = "$ProjectDir\models"

Write-Host ""
Write-Host "=== KKR ML Retrain Health Check ===" -ForegroundColor Cyan
Write-Host (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Write-Host ""

# 1. Task Scheduler status
Write-Host "-- Task Scheduler --" -ForegroundColor Yellow
try {
    $task = Get-ScheduledTask -TaskName "KKR_ML_FullRetrain" -ErrorAction Stop
    $info = Get-ScheduledTaskInfo -TaskName "KKR_ML_FullRetrain"
    Write-Host "  Status      : $($task.State)"
    Write-Host "  Logon Type  : $($task.Principal.LogonType)"
    Write-Host "  User        : $($task.Principal.UserId)"
    Write-Host "  Last Run    : $($info.LastRunTime)"
    if ($info.LastTaskResult -eq 0) {
        Write-Host "  Last Result : $($info.LastTaskResult)  SUCCESS" -ForegroundColor Green
    } else {
        Write-Host "  Last Result : $($info.LastTaskResult)  FAILED" -ForegroundColor Red
    }
    Write-Host "  Next Run    : $($info.NextRunTime)"
} catch {
    Write-Host "  Task not found -- re-register it." -ForegroundColor Red
}

# 2. Lock file check
Write-Host ""
Write-Host "-- Lock File --" -ForegroundColor Yellow
if (Test-Path $LockFile) {
    $age = (Get-Date) - (Get-Item $LockFile).LastWriteTime
    $ageH = [int]$age.TotalHours
    Write-Host "  WARN: lock file exists (age: ${ageH}h)" -ForegroundColor Red
    Write-Host "  If no python running, delete: $LockFile"
} else {
    Write-Host "  OK: no lock file" -ForegroundColor Green
}

# 3. Model files freshness
Write-Host ""
Write-Host "-- Model Files --" -ForegroundColor Yellow
$expected = @(
    'hmm_params.json', 'hmm_posteriors.json',
    'lgbm_model.txt', 'lgbm_regime_0.txt', 'lgbm_regime_1.txt',
    'lgbm_regime_2.txt', 'lgbm_regime_3.txt',
    'meta_lgbm.txt', 'stacking_weights.json',
    'shap_gate_weights.json', 'conformal_scores.json',
    'lstm_model.pt', 'xgb_phase2.json',
    'ppo_policy.zip', 'ppo_policy_weights.json', 'signal_dataset.csv', 'sector_data.pkl'
)
foreach ($f in $expected) {
    $fp = "$ModelsDir\$f"
    if (Test-Path $fp) {
        $ageDays = [int]((Get-Date) - (Get-Item $fp).LastWriteTime).TotalDays
        if ($ageDays -gt 45) {
            Write-Host "  [OK-OLD] $f  ($ageDays days old)" -ForegroundColor Yellow
        } else {
            Write-Host "  [OK]     $f  ($ageDays days old)" -ForegroundColor Green
        }
    } else {
        Write-Host "  [MISS]   $f" -ForegroundColor Red
    }
}

# 4. Last log tail
Write-Host ""
Write-Host "-- Last 15 Log Lines --" -ForegroundColor Yellow
if (Test-Path $LogFile) {
    Get-Content $LogFile -Tail 15
} else {
    Write-Host "  No log file yet." -ForegroundColor Red
}

Write-Host ""
Write-Host "=== End Health Check ===" -ForegroundColor Cyan
Write-Host ""
