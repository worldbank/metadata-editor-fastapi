@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: Metadata Editor FastAPI - Start Script (Windows)
:: ============================================================

:: Default configuration
set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "VENV_DIR=%PROJECT_DIR%\.venv"
set "MAIN_FILE=%PROJECT_DIR%\main.py"
set "PID_FILE=%PROJECT_DIR%\logs\app.pid"
set "LOG_FILE=%PROJECT_DIR%\logs\app.log"
set "LOG_ERR_FILE=%PROJECT_DIR%\logs\app_err.log"
set "DEFAULT_HOST=0.0.0.0"
set "DEFAULT_PORT=8000"

:: Allow override via environment variables
if "%HOST%"=="" set "HOST=%DEFAULT_HOST%"
if "%PORT%"=="" set "PORT=%DEFAULT_PORT%"
if "%CONDA_ENV_NAME%"=="" set "CONDA_ENV_NAME=metadata-editor"

:: Internal state
set "PYTHON_EXEC="
set "ENV_SOURCE="

:: ============================================================
:: Parse arguments
:: ============================================================
if /I "%~1"=="--help"  goto :show_help
if /I "%~1"=="-h"      goto :show_help
if /I "%~1"=="--check" goto :run_checks

if not "%~1"=="" (
    echo [ERROR] Unknown option: %~1
    echo [ERROR] Use --help for usage information
    exit /b 1
)

goto :main

:: ============================================================
:show_help
:: ============================================================
echo Metadata Editor FastAPI - Start Script (Windows)
echo.
echo Usage: %~nx0 [options]
echo.
echo Options:
echo   --help, -h    Show this help message
echo   --check       Run checks only without starting the application
echo.
echo Environment variables:
echo   HOST              Server host (default: 0.0.0.0)
echo   PORT              Server port (default: 8000)
echo   STORAGE_PATH      Path to data storage directory
echo   CONDA_ENV_NAME    Conda environment name to use (default: metadata-editor)
echo.
echo Python Environment Detection (in priority order):
echo   1. Conda environment named 'metadata-editor' (or %%CONDA_ENV_NAME%%)
echo   2. Currently active conda environment (CONDA_DEFAULT_ENV is set)
echo   3. Virtual environment (.venv\) if present
echo   4. System Python
echo.
echo Examples:
echo   %~nx0                              Start with default settings
echo   set HOST=127.0.0.1 ^& %~nx0       Start on localhost only
echo   set PORT=8000 ^& %~nx0            Start on port 8000
echo   set CONDA_ENV_NAME=myenv ^& %~nx0 Use a custom conda environment name
echo.
goto :eof

:: ============================================================
:create_directories
:: ============================================================
if not exist "%PROJECT_DIR%\logs" (
    mkdir "%PROJECT_DIR%\logs" >nul 2>&1
    echo [INFO] Created logs directory
)
if not exist "%PROJECT_DIR%\jobs" (
    mkdir "%PROJECT_DIR%\jobs" >nul 2>&1
    echo [INFO] Created jobs directory
)
goto :eof

:: ============================================================
:detect_python
:: ============================================================
echo [INFO] Detecting Python environment...

:: --- 1. Try named conda environment ---
where conda >nul 2>&1
if !errorlevel! equ 0 (
    echo [INFO] Conda found. Checking for environment: %CONDA_ENV_NAME%

    conda env list 2>nul | findstr /C:"%CONDA_ENV_NAME%" >nul 2>&1
    if !errorlevel! equ 0 (
        echo [INFO] Found conda environment: %CONDA_ENV_NAME%
        :: Resolve the actual python.exe path inside the conda env
        for /f "delims=" %%P in ('conda run -n %CONDA_ENV_NAME% python -c "import sys, os; print(os.path.normpath(sys.executable))" 2^>nul') do (
            set "PYTHON_EXEC=%%P"
        )
        if not "!PYTHON_EXEC!"=="" (
            echo [SUCCESS] Using conda environment '%CONDA_ENV_NAME%'
            echo [INFO] Python: !PYTHON_EXEC!
            set "ENV_SOURCE=conda:%CONDA_ENV_NAME%"
            goto :python_found
        ) else (
            echo [WARNING] Could not resolve Python path in conda env '%CONDA_ENV_NAME%'
        )
    ) else (
        echo [WARNING] Conda environment '%CONDA_ENV_NAME%' not found
    )

    :: --- 2. Try active conda environment ---
    if not "%CONDA_DEFAULT_ENV%"=="" (
        echo [INFO] Active conda environment detected: %CONDA_DEFAULT_ENV%
        for /f "delims=" %%P in ('python -c "import sys, os; print(os.path.normpath(sys.executable))" 2^>nul') do (
            set "PYTHON_EXEC=%%P"
        )
        if not "!PYTHON_EXEC!"=="" (
            "!PYTHON_EXEC!" -c "import uvicorn" >nul 2>&1
            if !errorlevel! equ 0 (
                echo [SUCCESS] Using active conda environment: %CONDA_DEFAULT_ENV%
                set "ENV_SOURCE=conda-active:%CONDA_DEFAULT_ENV%"
                goto :python_found
            ) else (
                echo [WARNING] uvicorn not found in active conda env '%CONDA_DEFAULT_ENV%'
            )
        )
    )
)

:: --- 3. Try .venv virtual environment ---
if exist "%VENV_DIR%\Scripts\python.exe" (
    set "PYTHON_EXEC=%VENV_DIR%\Scripts\python.exe"
    "%VENV_DIR%\Scripts\python.exe" -c "import uvicorn" >nul 2>&1
    if !errorlevel! equ 0 (
        echo [SUCCESS] Using virtual environment: %VENV_DIR%
        set "ENV_SOURCE=venv"
        goto :python_found
    ) else (
        echo [WARNING] uvicorn not found in .venv - run: %VENV_DIR%\Scripts\pip install -r requirements.txt
        set "PYTHON_EXEC="
    )
)

:: --- 4. Try system Python ---
for %%P in (python python3) do (
    where %%P >nul 2>&1
    if !errorlevel! equ 0 (
        %%P -c "import uvicorn" >nul 2>&1
        if !errorlevel! equ 0 (
            for /f "delims=" %%Q in ('%%P -c "import sys, os; print(os.path.normpath(sys.executable))" 2^>nul') do (
                set "PYTHON_EXEC=%%Q"
            )
            echo [SUCCESS] Using system Python: !PYTHON_EXEC!
            set "ENV_SOURCE=system"
            goto :python_found
        )
    )
)

echo [ERROR] Could not find Python with uvicorn installed.
echo [ERROR] Please set up an environment using one of these options:
echo.
echo [ERROR] Option A - Conda (recommended for geospatial support):
echo [ERROR]   conda create -n metadata-editor python=3.11 -y
echo [ERROR]   conda activate metadata-editor
echo [ERROR]   conda install -c conda-forge gdal fiona geopandas rasterio pyproj shapely -y
echo [ERROR]   pip install -r requirements.txt
echo.
echo [ERROR] Option B - Virtual environment:
echo [ERROR]   python -m venv .venv
echo [ERROR]   .venv\Scripts\activate
echo [ERROR]   pip install -r requirements.txt
exit /b 1

:python_found
goto :eof

:: ============================================================
:check_dependencies
:: ============================================================
echo [INFO] Checking dependencies...
if not exist "%MAIN_FILE%" (
    echo [ERROR] Main application file not found: %MAIN_FILE%
    exit /b 1
)
"%PYTHON_EXEC%" -c "import uvicorn, fastapi" >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Required packages (uvicorn, fastapi) not found in: %PYTHON_EXEC%
    echo [ERROR] Run: pip install -r requirements.txt
    exit /b 1
)
echo [SUCCESS] All dependencies are available
goto :eof

:: ============================================================
:check_env_config
:: ============================================================
echo [INFO] Checking environment configuration...
if exist "%PROJECT_DIR%\.env" (
    echo [SUCCESS] Found .env configuration file
) else (
    echo [WARNING] No .env file found - using default configuration
)
if not "%STORAGE_PATH%"=="" (
    if not exist "%STORAGE_PATH%" (
        echo [ERROR] STORAGE_PATH directory does not exist: %STORAGE_PATH%
        echo [ERROR] Please create the directory or update your .env file
        exit /b 1
    )
    echo [SUCCESS] STORAGE_PATH is valid: %STORAGE_PATH%
) else (
    echo [WARNING] STORAGE_PATH not set - path validation is disabled
)
goto :eof

:: ============================================================
:is_app_running
:: ============================================================
if exist "%PID_FILE%" (
    set /p EXISTING_PID=<"%PID_FILE%"
    if "!EXISTING_PID!"=="" (
        del /f "%PID_FILE%" >nul 2>&1
        exit /b 1
    )
    tasklist /fi "PID eq !EXISTING_PID!" 2>nul | findstr /C:"!EXISTING_PID!" >nul 2>&1
    if !errorlevel! equ 0 (
        exit /b 0
    ) else (
        del /f "%PID_FILE%" >nul 2>&1
        exit /b 1
    )
)
exit /b 1

:: ============================================================
:start_app
:: ============================================================
echo [INFO] Starting Metadata Editor FastAPI application...
echo [INFO]   Host:        %HOST%
echo [INFO]   Port:        %PORT%
echo [INFO]   Python:      %PYTHON_EXEC%
echo [INFO]   Environment: %ENV_SOURCE%
echo [INFO]   Log file:    %LOG_FILE%

:: Use PowerShell Start-Process to launch uvicorn in the background and capture PID.
:: RedirectStandardOutput / RedirectStandardError require absolute paths.
powershell -NoProfile -Command ^
    "$p = Start-Process -FilePath '%PYTHON_EXEC%' ^
        -ArgumentList '-m','uvicorn','main:app','--host','%HOST%','--port','%PORT%','--log-level','info' ^
        -WorkingDirectory '%PROJECT_DIR%' ^
        -RedirectStandardOutput '%LOG_FILE%' ^
        -RedirectStandardError '%LOG_ERR_FILE%' ^
        -WindowStyle Hidden ^
        -PassThru; ^
    $p.Id | Out-File -FilePath '%PID_FILE%' -Encoding ascii -NoNewline"

if not exist "%PID_FILE%" (
    echo [ERROR] Failed to start application or capture PID
    echo [ERROR] Check log for details: %LOG_FILE%
    exit /b 1
)

:: Wait briefly for the process to initialise
echo [INFO] Waiting for application to initialise...
timeout /t 3 /nobreak >nul

set /p APP_PID=<"%PID_FILE%"
tasklist /fi "PID eq %APP_PID%" 2>nul | findstr /C:"%APP_PID%" >nul 2>&1
if !errorlevel! equ 0 (
    echo.
    echo [SUCCESS] Application started successfully^^!
    echo [SUCCESS]   PID:               %APP_PID%
    echo [SUCCESS]   URL:               http://%HOST%:%PORT%
    echo [SUCCESS]   API Documentation: http://%HOST%:%PORT%/docs
    echo [SUCCESS]   Log file:          %LOG_FILE%
    echo.
    echo [INFO] To stop the application run: stop.bat
) else (
    echo [ERROR] Application failed to start. Check logs:
    echo [ERROR]   %LOG_FILE%
    echo [ERROR]   %LOG_ERR_FILE%
    del /f "%PID_FILE%" >nul 2>&1
    exit /b 1
)
goto :eof

:: ============================================================
:run_checks
:: ============================================================
echo [INFO] === Running checks only ===
call :create_directories
call :detect_python
if !errorlevel! neq 0 exit /b 1
call :check_dependencies
if !errorlevel! neq 0 exit /b 1
call :check_env_config
echo [SUCCESS] All checks passed^^!
goto :eof

:: ============================================================
:main
:: ============================================================
echo [INFO] === Metadata Editor FastAPI - Start Script (Windows) ===
echo [INFO] Project directory: %PROJECT_DIR%

call :is_app_running
if !errorlevel! equ 0 (
    echo [WARNING] Application is already running (PID: %EXISTING_PID%)
    echo [WARNING] Use stop.bat to stop it first
    exit /b 1
)

call :create_directories

call :detect_python
if !errorlevel! neq 0 exit /b 1

call :check_dependencies
if !errorlevel! neq 0 exit /b 1

call :check_env_config

call :start_app
if !errorlevel! neq 0 exit /b 1

echo [SUCCESS] === Application startup completed ===
endlocal
