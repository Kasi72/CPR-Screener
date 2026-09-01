# auto_retrain_catchup.ps1 — Fire on logon. Retrain if monthly run was missed.

$STAMP   = "D:\Claude code\nse-screener\auto_retrain_cpr.last_run"
$MAIN    = "D:\Claude code\nse-screener\scripts\auto_retrain_cpr.ps1"
$PYTHON  = "C:\Users\drkkr\AppData\Local\Programs\Python\Python310\python.exe"

$threshold = 28  # days — if last run older than this, trigger retrain

if (Test-Path $STAMP) {
    $lastRun = [datetime]::Parse((Get-Content $STAMP))
    $age = (Get-Date) - $lastRun
    if ($age.TotalDays -lt $threshold) {
        Write-Host "Last retrain $([int]$age.TotalDays)d ago — within $threshold day window. Skip."
        exit 0
    }
    Write-Host "Last retrain $([int]$age.TotalDays)d ago — overdue. Triggering catchup run."
} else {
    Write-Host "No last-run stamp found — triggering initial retrain."
}

# Launch main script in background so logon completes normally
Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-ExecutionPolicy Bypass -NonInteractive -File `"$MAIN`"" `
    -WindowStyle Hidden
