@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: Metadata Editor FastAPI - Stop Script (Windows)
:: ============================================================

set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "PID_FILE=%PROJECT_DIR%\logs\app.pid"
set "LOG_FILE=%PROJECT_DIR%\logs\app.log"

:: ============================================================
:: Parse arguments
:: ============================================================
if /I "%~1"=="--help"   goto :show_help
if /I "%~1"=="-h"       goto :show_help
if /I "%~1"=="--force"  goto :force_stop
if /I "%~1"=="--status" goto :status

if not "%~1"=="" (
    echo [ERROR] Unknown option: %~1
    echo [ERROR] Use --help for usage information
    exit /b 1
)

goto :main

:: ============================================================
:show_help
:: ============================================================
echo Metadata Editor FastAPI - Stop Script (Windows)
echo.
echo Usage: %~nx0 [options]
echo.
echo Options:
echo   --help, -h    Show this help message
echo   --force       Force kill all Python uvicorn processes for this app
echo   --status      Check if the application is currently running
echo.
echo Examples:
echo   %~nx0           Stop application gracefully
echo   %~nx0 --force   Force stop if graceful stop fails
echo   %~nx0 --status  Check status
echo.
goto :eof

:: ============================================================
:is_app_running
:: ============================================================
if exist "%PID_FILE%" (
    set /p APP_PID=<"%PID_FILE%"
    if "!APP_PID!"=="" (
        del /f "%PID_FILE%" >nul 2>&1
        exit /b 1
    )
    tasklist /fi "PID eq !APP_PID!" 2>nul | findstr /C:"!APP_PID!" >nul 2>&1
    if !errorlevel! equ 0 (
        exit /b 0
    ) else (
        del /f "%PID_FILE%" >nul 2>&1
        exit /b 1
    )
)
exit /b 1

:: ============================================================
:stop_by_pid
:: ============================================================
set /p APP_PID=<"%PID_FILE%"
echo [INFO] Stopping application (PID: %APP_PID%)...

:: Graceful stop (taskkill without /F sends WM_CLOSE / CTRL_C_EVENT first)
taskkill /PID %APP_PID% >nul 2>&1
if !errorlevel! neq 0 (
    echo [WARNING] Graceful stop signal failed, trying force kill...
    taskkill /F /PID %APP_PID% >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to stop process %APP_PID%
        exit /b 1
    )
)

:: Wait for the process to exit (up to 15 seconds)
set /a COUNT=0
:wait_loop
tasklist /fi "PID eq %APP_PID%" 2>nul | findstr /C:"%APP_PID%" >nul 2>&1
if !errorlevel! neq 0 goto :process_stopped
set /a COUNT+=1
if !COUNT! geq 15 (
    echo [WARNING] Process did not exit after 15 seconds, sending force kill...
    taskkill /F /T /PID %APP_PID% >nul 2>&1
    goto :process_stopped
)
timeout /t 1 /nobreak >nul
goto :wait_loop

:process_stopped
del /f "%PID_FILE%" >nul 2>&1
echo [SUCCESS] Application stopped
goto :eof

:: ============================================================
:force_stop
:: ============================================================
echo [INFO] === Force stopping uvicorn processes ===

:: Find Python processes running uvicorn main:app
set "FOUND=0"
for /f "tokens=2" %%P in ('wmic process where "commandline like '%%uvicorn%%main:app%%'" get processid /format:list 2^>nul ^| findstr /C:"ProcessId="') do (
    set "KILL_PID=%%P"
    if not "!KILL_PID!"=="" (
        echo [INFO] Stopping process !KILL_PID!...
        taskkill /F /T /PID !KILL_PID! >nul 2>&1
        if !errorlevel! equ 0 (
            echo [SUCCESS] Stopped process !KILL_PID!
        ) else (
            echo [WARNING] Could not stop process !KILL_PID! (may have already exited)
        )
        set "FOUND=1"
    )
)

if "%FOUND%"=="0" (
    echo [WARNING] No running uvicorn processes found
)

if exist "%PID_FILE%" del /f "%PID_FILE%" >nul 2>&1

echo [SUCCESS] Force stop completed
goto :eof

:: ============================================================
:status
:: ============================================================
echo [INFO] === Application Status ===
call :is_app_running
if !errorlevel! equ 0 (
    echo [SUCCESS] Application is running (PID: %APP_PID%)
    :: Show process command line
    wmic process where "processid='%APP_PID%'" get commandline,processid /format:list 2>nul | findstr /V "^$"
) else (
    echo [WARNING] Application is not running

    :: Check for any lingering uvicorn processes
    set "STRAY="
    for /f "tokens=2" %%P in ('wmic process where "commandline like '%%uvicorn%%main:app%%'" get processid /format:list 2^>nul ^| findstr /C:"ProcessId="') do (
        if not "%%P"=="" set "STRAY=%%P"
    )
    if not "!STRAY!"=="" (
        echo [WARNING] Found uvicorn process(es) not tracked by PID file: !STRAY!
        echo [WARNING] Use --force to stop them
    )
)
goto :eof

:: ============================================================
:main
:: ============================================================
echo [INFO] === Metadata Editor FastAPI - Stop Script (Windows) ===
echo [INFO] Project directory: %PROJECT_DIR%

call :is_app_running
if !errorlevel! neq 0 (
    echo [WARNING] Application is not running

    :: Still check for any stray uvicorn processes
    set "STRAY="
    for /f "tokens=2" %%P in ('wmic process where "commandline like '%%uvicorn%%main:app%%'" get processid /format:list 2^>nul ^| findstr /C:"ProcessId="') do (
        if not "%%P"=="" set "STRAY=%%P"
    )
    if not "!STRAY!"=="" (
        echo [WARNING] Found uvicorn process(es) not tracked by PID file: !STRAY!
        echo [WARNING] Use --force to stop them
    )
    exit /b 0
)

call :stop_by_pid
if !errorlevel! neq 0 (
    echo [ERROR] Failed to stop application via PID file
    echo [ERROR] Try: %~nx0 --force
    exit /b 1
)

echo [SUCCESS] === Application shutdown completed ===
endlocal
