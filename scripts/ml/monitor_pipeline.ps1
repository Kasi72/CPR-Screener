# monitor_pipeline.ps1
# Live Kaggle kernel status poller - run in a second PS window while pipeline runs.
# Polls Phase 2c then Phase 3 kernel every 60s and shows elapsed time + AUC when done.
#
# Usage: .\scripts\ml\monitor_pipeline.ps1

param(
    [ValidateSet('phase2c','phase3','both')]
    [string]$Watch = 'both',
    [int]$PollSec = 60
)

$REPO = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent

function KernelStatus([string]$slug) {
    $r = & kaggle kernels status $slug 2>&1
    $s = ($r -join ' ').ToLower()
    if ($s -match 'complete')           { return 'COMPLETE' }
    if ($s -match 'error')              { return 'ERROR' }
    if ($s -match 'running')            { return 'RUNNING' }
    if ($s -match 'queued|pending')     { return 'QUEUED' }
    if ($s -match 'cancel')             { return 'CANCELLED' }
    return "UNKNOWN: $($r -join ' ')"
}

function ShowMetrics([string]$path, [string]$label) {
    if (-not (Test-Path $path)) { return }
    $m = Get-Content $path | ConvertFrom-Json
    Write-Host "  ── $label ──" -ForegroundColor Cyan
    $m.PSObject.Properties | Where-Object { $_.Value -is [double] -or $_.Value -is [float] } |
        ForEach-Object { Write-Host ("  {0,-25} {1}" -f $_.Name, $_.Value) }
}

function MonitorKernel([string]$slug, [string]$label, [string]$metricsFile) {
    Write-Host ""
    Write-Host "Watching: $slug" -ForegroundColor Cyan
    $start = Get-Date
    $last  = ""

    while ($true) {
        $status  = KernelStatus $slug
        $elapsed = [Math]::Round(((Get-Date) - $start).TotalMinutes, 1)
        $ts      = (Get-Date).ToString('HH:mm:ss')

        if ($status -ne $last) {
            Write-Host "[$ts] ${elapsed}min  $label  →  $status" -ForegroundColor White
            $last = $status
        } else {
            Write-Host "[$ts] ${elapsed}min  $label  →  $status" -ForegroundColor DarkGray
        }

        if ($status -eq 'COMPLETE') {
            Write-Host "[$ts] $label DONE in ${elapsed}min" -ForegroundColor Green
            if ($metricsFile) {
                $mpath = Join-Path $REPO "models\$metricsFile"
                ShowMetrics $mpath $label
            }
            return $true
        }
        if ($status -in @('ERROR','CANCELLED')) {
            Write-Host "[$ts] $label FAILED: $status" -ForegroundColor Red
            return $false
        }

        Start-Sleep $PollSec
    }
}

Set-Location $REPO

Write-Host "Pipeline Monitor - watching Kaggle kernels" -ForegroundColor Cyan
Write-Host "Poll interval: ${PollSec}s  |  Ctrl+C to stop" -ForegroundColor DarkGray

if ($Watch -in @('phase2c','both')) {
    $ok = MonitorKernel 'drkasi/cpr-phase-2c-lgbm-signal-scorer' 'Phase2c' 'phase2c_metrics.json'
    if (-not $ok -and $Watch -eq 'both') {
        Write-Host "Phase 2c failed - not waiting for Phase 3." -ForegroundColor Red
        exit 1
    }
}

if ($Watch -in @('phase3','both')) {
    MonitorKernel 'drkasi/cpr-phase-3-lstm-training' 'Phase3' 'phase3_metrics.json'
}

Write-Host ""
Write-Host "Monitor done." -ForegroundColor Cyan

