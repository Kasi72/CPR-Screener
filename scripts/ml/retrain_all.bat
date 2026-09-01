@echo off
setlocal EnableDelayedExpansion

:: ─── Config ────────────────────────────────────────────────────────────────
set PYTHON=C:\Users\drkkr\AppData\Local\Programs\Python\Python310\python.exe
set PROJECT=D:\Claude code\nse-screener
set SCRIPT=scripts\ml\run_all.py
set LOGFILE=%PROJECT%\scripts\ml\retrain_all.log
set LOCKFILE=%PROJECT%\scripts\ml\retrain.lock

:: ─── Timestamp ──────────────────────────────────────────────────────────────
for /f "tokens=1-2 delims= " %%a in ('wmic os get LocalDateTime /value ^| find "="') do set DT=%%b
set TS=%DT:~0,4%-%DT:~4,2%-%DT:~6,2% %DT:~8,2%:%DT:~10,2%:%DT:~12,2%

:: ─── Lock file — prevent concurrent runs ───────────────────────────────────
if exist "%LOCKFILE%" (
    echo [%TS%] SKIP: lock file exists — previous run still active or crashed >> "%LOGFILE%"
    echo Check and delete %LOCKFILE% manually if previous run is dead. >> "%LOGFILE%"
    exit /b 1
)
echo %DATE% %TIME% > "%LOCKFILE%"

:: ─── Log header ─────────────────────────────────────────────────────────────
echo. >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"
echo [%TS%] KKR_ML_FullRetrain START >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

:: ─── Verify Python exists ────────────────────────────────────────────────────
if not exist "%PYTHON%" (
    echo [%TS%] ERROR: Python not found at %PYTHON% >> "%LOGFILE%"
    del "%LOCKFILE%"
    exit /b 2
)

:: ─── Verify project dir ──────────────────────────────────────────────────────
if not exist "%PROJECT%" (
    echo [%TS%] ERROR: Project dir not found: %PROJECT% >> "%LOGFILE%"
    del "%LOCKFILE%"
    exit /b 3
)

:: ─── Step 0a: Update sector index data (incremental, ~5 min) ─────────────────
cd /d "%PROJECT%"
echo [%TS%] Updating sector index data... >> "%LOGFILE%"
"%PYTHON%" scripts\ml\sector_features.py >> "%LOGFILE%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%TS%] WARN: sector_features.py failed -- continuing with cached data >> "%LOGFILE%"
)

:: ─── Step 0b: Update NSE bhavcopy delivery data (incremental, picks up from last date) ──
echo [%TS%] Updating bhavcopy delivery data... >> "%LOGFILE%"
"%PYTHON%" scripts\ml\download_bhavcopy.py >> "%LOGFILE%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%TS%] WARN: download_bhavcopy.py failed -- continuing with cached data >> "%LOGFILE%"
)

:: ─── Step 1+: Run full training pipeline ─────────────────────────────────────
echo [%TS%] Starting ML training pipeline... >> "%LOGFILE%"
"%PYTHON%" "%SCRIPT%" >> "%LOGFILE%" 2>&1
set EXITCODE=%ERRORLEVEL%

:: ─── Timestamp end ───────────────────────────────────────────────────────────
for /f "tokens=1-2 delims= " %%a in ('wmic os get LocalDateTime /value ^| find "="') do set DT=%%b
set TS=%DT:~0,4%-%DT:~4,2%-%DT:~6,2% %DT:~8,2%:%DT:~10,2%:%DT:~12,2%

:: ─── Result ──────────────────────────────────────────────────────────────────
if %EXITCODE% EQU 0 (
    echo [%TS%] SUCCESS: All phases complete. Exit code 0 >> "%LOGFILE%"
) else (
    echo [%TS%] FAILED: Exit code %EXITCODE% >> "%LOGFILE%"
    echo [%TS%] Check log above for error details >> "%LOGFILE%"
)

echo ============================================================ >> "%LOGFILE%"

:: ─── Remove lock ─────────────────────────────────────────────────────────────
del "%LOCKFILE%"

exit /b %EXITCODE%
