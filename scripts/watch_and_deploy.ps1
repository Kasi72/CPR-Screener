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

if ($p4error -or $p2berror) {
    Write-Host "One or more kernels failed. Check logs manually."
    Write-Host "  Phase 4 : $P4LOG"
    Write-Host "  Phase 2b: $P2BLOG"
    exit 1
}

Write-Host "Both kernels complete. Running deploy..."
& pwsh $DEPLOY
