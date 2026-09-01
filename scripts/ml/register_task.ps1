# register_task.ps1 -- run as Administrator
# Uses Task Scheduler COM API -- handles spaces in paths, monthly trigger, SYSTEM account

$taskName = "KKR_ML_FullRetrain"
$batFile  = "D:\Claude code\nse-screener\scripts\ml\retrain_all.bat"

# Remove existing
try { (New-Object -ComObject "Schedule.Service" | ForEach-Object { $_.Connect(); $_.GetFolder("\").DeleteTask($taskName, 0) }) } catch {}

$sched = New-Object -ComObject "Schedule.Service"
$sched.Connect()
$root = $sched.GetFolder("\")
$def  = $sched.NewTask(0)

# --- Action: cmd /c "<bat>" ---
$act           = $def.Actions.Create(0)   # TASK_ACTION_EXEC
$act.Path      = "cmd.exe"
$act.Arguments = "/c `"$batFile`""

# --- Monthly trigger: day 1, 02:00, all months ---
$trig                 = $def.Triggers.Create(4)   # TASK_TRIGGER_MONTHLY
$trig.StartBoundary   = "2026-09-01T02:00:00"
$trig.MonthsOfYear    = 4095                        # all 12 months (bitmask)
$trig.DaysOfMonth     = 1                            # bit 0 = day 1
$trig.Enabled         = $true

# --- Principal: SYSTEM, highest ---
$def.Principal.UserId   = "SYSTEM"
$def.Principal.LogonType = 5   # TASK_LOGON_SERVICE_ACCOUNT
$def.Principal.RunLevel  = 1   # TASK_RUNLEVEL_HIGHEST

# --- Settings ---
$def.Settings.StartWhenAvailable    = $true
$def.Settings.ExecutionTimeLimit    = "PT8H"
$def.Settings.MultipleInstances     = 3     # TASK_INSTANCES_IGNORE_NEW
$def.Settings.DisallowStartIfOnBatteries = $false
$def.Settings.StopIfGoingOnBatteries    = $false

# --- Register: 6=CREATE_OR_UPDATE, 5=TASK_LOGON_SERVICE_ACCOUNT ---
$root.RegisterTaskDefinition($taskName, $def, 6, "SYSTEM", $null, 5) | Out-Null

Write-Host "Registered. Verifying..." -ForegroundColor Cyan
$t    = Get-ScheduledTask    -TaskName $taskName
$info = Get-ScheduledTaskInfo -TaskName $taskName

Write-Host "  LogonType  : $($t.Principal.LogonType)"
Write-Host "  Next Run   : $($info.NextRunTime)"
Write-Host "  StartIfMiss: $($t.Settings.StartWhenAvailable)"

if ($t.Principal.LogonType -eq "ServiceAccount" -and $info.NextRunTime) {
    Write-Host "SUCCESS: Runs as SYSTEM without login on 1st of each month at 02:00." -ForegroundColor Green
} else {
    Write-Host "WARN: Next Run=$($info.NextRunTime)  LogonType=$($t.Principal.LogonType)" -ForegroundColor Yellow
}
