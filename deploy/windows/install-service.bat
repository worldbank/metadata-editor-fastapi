@echo off
setlocal EnableDelayedExpansion

:: =============================================================
:: install-service.bat
::
:: Installs editor-fastapi as a Windows service using NSSM.
:: Must be run from an elevated (Administrator) Command Prompt.
::
:: REQUIRED - set before running:
::   CONDA_PYTHON_PATH   Absolute path to python.exe in the conda
::                       environment. Find it with:
::                         conda activate metadata-editor
::                         where python
::
:: OPTIONAL overrides (defaults shown):
::   PROJECT_DIR         Application root - where main.py lives
::                       (default: parent of the deploy\windows folder)
::   HOST                Bind address          (default: 0.0.0.0)
::   PORT                Listening port        (default: 8000)
::   NSSM_PATH           Full path to nssm.exe (default: looks on PATH)
::   SHARED_GROUP        Local group for shared storage access
::                       (default: editor-shared)
::   WEB_SERVER_USER     IIS/Apache account to add to shared group
::                       (default: IIS_IUSRS)
::   STORAGE_PATH        Shared folder path - permissions set if provided
::
:: Usage (from an Administrator Command Prompt):
::   set CONDA_PYTHON_PATH=C:\Users\myuser\miniconda3\envs\metadata-editor\python.exe
::   set STORAGE_PATH=C:\inetpub\wwwroot\metadata-editor\datafiles
::   install-service.bat
::
::   install-service.bat --uninstall
::   install-service.bat --status
:: =============================================================

set "SERVICE_NAME=editor-fastapi"
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

:: Default PROJECT_DIR = two levels up from deploy\windows\
for %%D in ("%SCRIPT_DIR%\..\..") do set "DEFAULT_PROJECT_DIR=%%~fD"
if "%PROJECT_DIR%"=="" set "PROJECT_DIR=%DEFAULT_PROJECT_DIR%"

if "%HOST%"==""           set "HOST=0.0.0.0"
if "%PORT%"==""           set "PORT=8000"
if "%SHARED_GROUP%"==""   set "SHARED_GROUP=editor-shared"
if "%WEB_SERVER_USER%"=="" set "WEB_SERVER_USER=IIS_IUSRS"

:: =============================================================
:: Argument handling
:: =============================================================
if /I "%~1"=="--uninstall" goto :uninstall_service
if /I "%~1"=="--status"    goto :status
if /I "%~1"=="--help"      goto :show_help
if /I "%~1"=="-h"          goto :show_help
if not "%~1"=="" (
    echo [ERROR] Unknown option: %~1
    echo [ERROR] Use --help for usage information
    exit /b 1
)
goto :main

:: =============================================================
:show_help
:: =============================================================
echo Metadata Editor FastAPI - Windows Service Installer (NSSM)
echo.
echo Usage: %~nx0 [--uninstall ^| --status ^| --help]
echo.
echo Required environment variables:
echo   CONDA_PYTHON_PATH   Full path to python.exe in the conda environment
echo.
echo Optional environment variables:
echo   PROJECT_DIR         Application root directory
echo   HOST                Bind address (default: 0.0.0.0)
echo   PORT                Port (default: 8000)
echo   NSSM_PATH           Path to nssm.exe (default: searches PATH)
echo   SHARED_GROUP        Local group for shared storage (default: editor-shared)
echo   WEB_SERVER_USER     Web server account for shared group (default: IIS_IUSRS)
echo   STORAGE_PATH        Shared folder to configure permissions on
echo.
echo Example:
echo   set CONDA_PYTHON_PATH=C:\Users\user\miniconda3\envs\metadata-editor\python.exe
echo   set STORAGE_PATH=C:\inetpub\wwwroot\metadata-editor\datafiles
echo   %~nx0
echo.
goto :eof

:: =============================================================
:check_admin
:: =============================================================
net session >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] This script must be run as Administrator.
    echo [ERROR] Right-click Command Prompt and choose "Run as administrator".
    exit /b 1
)
goto :eof

:: =============================================================
:find_nssm
:: =============================================================
if not "%NSSM_PATH%"=="" (
    if not exist "%NSSM_PATH%" (
        echo [ERROR] NSSM_PATH does not exist: %NSSM_PATH%
        exit /b 1
    )
    echo [INFO] Using NSSM: %NSSM_PATH%
    goto :eof
)

:: Search PATH
where nssm >nul 2>&1
if !errorlevel! equ 0 (
    for /f "delims=" %%P in ('where nssm') do set "NSSM_PATH=%%P"
    echo [INFO] Found NSSM on PATH: !NSSM_PATH!
    goto :eof
)

:: Common install locations
for %%P in (
    "C:\nssm\win64\nssm.exe"
    "C:\nssm\win32\nssm.exe"
    "C:\tools\nssm\win64\nssm.exe"
    "C:\Program Files\nssm\nssm.exe"
    "C:\ProgramData\chocolatey\bin\nssm.exe"
) do (
    if exist %%P (
        set "NSSM_PATH=%%~P"
        echo [INFO] Found NSSM at: !NSSM_PATH!
        goto :eof
    )
)

echo [ERROR] NSSM not found. Install it first:
echo.
echo   Option A - Chocolatey (recommended):
echo     choco install nssm
echo.
echo   Option B - Manual download:
echo     https://nssm.cc/download
echo     Extract nssm.exe and set NSSM_PATH to its location:
echo     set NSSM_PATH=C:\tools\nssm\win64\nssm.exe
echo.
exit /b 1

:: =============================================================
:validate_inputs
:: =============================================================
if "%CONDA_PYTHON_PATH%"=="" (
    echo [ERROR] CONDA_PYTHON_PATH is not set.
    echo.
    echo Find the correct path by running (in your conda environment^):
    echo   conda activate metadata-editor
    echo   where python
    echo.
    echo Then set it and re-run:
    echo   set CONDA_PYTHON_PATH=C:\path\to\python.exe
    echo   %~nx0
    exit /b 1
)

if not exist "%CONDA_PYTHON_PATH%" (
    echo [ERROR] CONDA_PYTHON_PATH does not exist: %CONDA_PYTHON_PATH%
    exit /b 1
)

:: Verify uvicorn is available
"%CONDA_PYTHON_PATH%" -c "import uvicorn" >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] uvicorn is not installed in the environment at:
    echo [ERROR]   %CONDA_PYTHON_PATH%
    echo [ERROR] Activate the conda environment and run:
    echo [ERROR]   pip install -r requirements.txt
    exit /b 1
)

if not exist "%PROJECT_DIR%\main.py" (
    echo [ERROR] main.py not found in PROJECT_DIR: %PROJECT_DIR%
    echo [ERROR] Set PROJECT_DIR to the application root directory.
    exit /b 1
)

:: Derive the conda env lib/bin directory for PATH injection
for %%D in ("%CONDA_PYTHON_PATH%") do set "CONDA_ENV_DIR=%%~dpD"
set "CONDA_ENV_DIR=%CONDA_ENV_DIR:~0,-1%"
set "CONDA_LIB_DIR=%CONDA_ENV_DIR%\Library\bin"

echo [INFO] Configuration:
echo [INFO]   CONDA_PYTHON_PATH : %CONDA_PYTHON_PATH%
echo [INFO]   CONDA_ENV_DIR     : %CONDA_ENV_DIR%
echo [INFO]   CONDA_LIB_DIR     : %CONDA_LIB_DIR%
echo [INFO]   PROJECT_DIR       : %PROJECT_DIR%
echo [INFO]   HOST:PORT         : %HOST%:%PORT%
echo [INFO]   SHARED_GROUP      : %SHARED_GROUP%
echo [INFO]   WEB_SERVER_USER   : %WEB_SERVER_USER%
if not "%STORAGE_PATH%"=="" (
    echo [INFO]   STORAGE_PATH      : %STORAGE_PATH%
) else (
    echo [INFO]   STORAGE_PATH      : (not set - skip permission setup)
)
goto :eof

:: =============================================================
:setup_shared_permissions
:: =============================================================
echo [INFO] Setting up shared group for web server access...

:: Create local group if it doesn't exist
net localgroup "%SHARED_GROUP%" >nul 2>&1
if !errorlevel! neq 0 (
    net localgroup "%SHARED_GROUP%" /add /comment:"Shared access group for editor-fastapi and web server"
    if !errorlevel! equ 0 (
        echo [SUCCESS] Created local group: %SHARED_GROUP%
    ) else (
        echo [WARNING] Could not create local group '%SHARED_GROUP%' - it may already exist
    )
) else (
    echo [INFO] Local group '%SHARED_GROUP%' already exists
)

:: The service runs as LOCAL SERVICE or NETWORK SERVICE by default under NSSM.
:: We add the web server account to the shared group for folder access.
:: Note: Windows services using a built-in account (LOCAL SERVICE, NETWORK SERVICE)
:: cannot be added to custom groups directly; use icacls on the folder instead.

:: Add web server user to the shared group
net localgroup "%SHARED_GROUP%" "%WEB_SERVER_USER%" /add >nul 2>&1
if !errorlevel! equ 0 (
    echo [SUCCESS] Added '%WEB_SERVER_USER%' to group '%SHARED_GROUP%'
) else (
    echo [WARNING] Could not add '%WEB_SERVER_USER%' to '%SHARED_GROUP%'
    echo [WARNING] The account may already be a member, or may not exist.
    echo [WARNING] Verify with: net localgroup %SHARED_GROUP%
)

:: Set permissions on the shared storage folder
if not "%STORAGE_PATH%"=="" (
    if not exist "%STORAGE_PATH%" (
        echo [ERROR] STORAGE_PATH does not exist: %STORAGE_PATH%
        exit /b 1
    )

    :: Grant full control to the shared group (inheritable)
    icacls "%STORAGE_PATH%" /grant "%SHARED_GROUP%:(OI)(CI)F" /T >nul 2>&1
    if !errorlevel! equ 0 (
        echo [SUCCESS] Granted full control to '%SHARED_GROUP%' on: %STORAGE_PATH%
    ) else (
        echo [WARNING] icacls failed for '%SHARED_GROUP%' - check permissions manually
    )

    :: Grant full control to NETWORK SERVICE (used by IIS app pools by default)
    icacls "%STORAGE_PATH%" /grant "NETWORK SERVICE:(OI)(CI)F" /T >nul 2>&1
    if !errorlevel! equ 0 (
        echo [SUCCESS] Granted full control to NETWORK SERVICE on: %STORAGE_PATH%
    )

    :: Ensure the service account (LOCAL SERVICE) also has access
    icacls "%STORAGE_PATH%" /grant "LOCAL SERVICE:(OI)(CI)F" /T >nul 2>&1
    echo [INFO] icacls summary for: %STORAGE_PATH%
    icacls "%STORAGE_PATH%"
) else (
    echo [WARNING] STORAGE_PATH not set - skipping directory permission setup.
    echo [WARNING] Set permissions manually after install:
    echo [WARNING]   icacls "C:\path\to\storage" /grant "%SHARED_GROUP%:(OI)(CI)F" /T
    echo [WARNING]   icacls "C:\path\to\storage" /grant "NETWORK SERVICE:(OI)(CI)F" /T
)
goto :eof

:: =============================================================
:create_directories
:: =============================================================
if not exist "%PROJECT_DIR%\logs" mkdir "%PROJECT_DIR%\logs" >nul 2>&1
if not exist "%PROJECT_DIR%\jobs" mkdir "%PROJECT_DIR%\jobs" >nul 2>&1
goto :eof

:: =============================================================
:install_service
:: =============================================================
echo [INFO] Installing service '%SERVICE_NAME%' via NSSM...

:: Remove any existing service with the same name
"%NSSM_PATH%" status "%SERVICE_NAME%" >nul 2>&1
if !errorlevel! equ 0 (
    echo [WARNING] Service '%SERVICE_NAME%' already exists - removing first...
    "%NSSM_PATH%" stop "%SERVICE_NAME%" >nul 2>&1
    "%NSSM_PATH%" remove "%SERVICE_NAME%" confirm >nul 2>&1
    echo [INFO] Existing service removed
)

:: Register the service
"%NSSM_PATH%" install "%SERVICE_NAME%" "%CONDA_PYTHON_PATH%"
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install service
    exit /b 1
)

:: Arguments passed to python.exe
"%NSSM_PATH%" set "%SERVICE_NAME%" AppParameters "-m uvicorn main:app --host %HOST% --port %PORT% --log-level info"

:: Working directory (project root)
"%NSSM_PATH%" set "%SERVICE_NAME%" AppDirectory "%PROJECT_DIR%"

:: Add conda env bin and Library\bin to PATH so GDAL/PROJ DLLs are found
"%NSSM_PATH%" set "%SERVICE_NAME%" AppEnvironmentExtra "PATH=%CONDA_ENV_DIR%;%CONDA_LIB_DIR%;%PATH%"

:: Stdout / stderr log files
"%NSSM_PATH%" set "%SERVICE_NAME%" AppStdout "%PROJECT_DIR%\logs\service.log"
"%NSSM_PATH%" set "%SERVICE_NAME%" AppStderr "%PROJECT_DIR%\logs\service_err.log"
"%NSSM_PATH%" set "%SERVICE_NAME%" AppStdoutCreationDisposition 4
"%NSSM_PATH%" set "%SERVICE_NAME%" AppStderrCreationDisposition 4

:: Rotate logs when they exceed 10 MB
"%NSSM_PATH%" set "%SERVICE_NAME%" AppRotateFiles 1
"%NSSM_PATH%" set "%SERVICE_NAME%" AppRotateBytes 10485760

:: Auto-start at boot
"%NSSM_PATH%" set "%SERVICE_NAME%" Start SERVICE_AUTO_START

:: Restart on failure with 10-second delay
"%NSSM_PATH%" set "%SERVICE_NAME%" AppRestartDelay 10000

:: Service description
"%NSSM_PATH%" set "%SERVICE_NAME%" Description "Metadata Editor FastAPI backend (editor-fastapi)"

echo [SUCCESS] Service configured

:: Start the service
echo [INFO] Starting service...
"%NSSM_PATH%" start "%SERVICE_NAME%"
if !errorlevel! neq 0 (
    echo [ERROR] Service failed to start. Check logs:
    echo [ERROR]   %PROJECT_DIR%\logs\service.log
    echo [ERROR]   %PROJECT_DIR%\logs\service_err.log
    echo [ERROR] Or use: sc query %SERVICE_NAME%
    exit /b 1
)

timeout /t 3 /nobreak >nul

sc query "%SERVICE_NAME%" | findstr /C:"RUNNING" >nul 2>&1
if !errorlevel! equ 0 (
    echo.
    echo [SUCCESS] Service is running^^!
    echo [SUCCESS]   URL:               http://%HOST%:%PORT%
    echo [SUCCESS]   API Documentation: http://%HOST%:%PORT%/docs
    echo [SUCCESS]   Log file:          %PROJECT_DIR%\logs\service.log
    echo.
    echo [INFO] Management commands:
    echo [INFO]   sc start   %SERVICE_NAME%
    echo [INFO]   sc stop    %SERVICE_NAME%
    echo [INFO]   sc query   %SERVICE_NAME%
    echo [INFO]   %NSSM_PATH% edit %SERVICE_NAME%     (GUI editor)
) else (
    echo [WARNING] Service may not be running. Check:
    echo [WARNING]   sc query %SERVICE_NAME%
    echo [WARNING]   %PROJECT_DIR%\logs\service_err.log
)
goto :eof

:: =============================================================
:uninstall_service
:: =============================================================
call :check_admin
echo [INFO] Uninstalling service '%SERVICE_NAME%'...

call :find_nssm
if !errorlevel! neq 0 exit /b 1

"%NSSM_PATH%" status "%SERVICE_NAME%" >nul 2>&1
if !errorlevel! neq 0 (
    echo [WARNING] Service '%SERVICE_NAME%' is not installed
    exit /b 0
)

"%NSSM_PATH%" stop "%SERVICE_NAME%" >nul 2>&1
echo [INFO] Service stopped
"%NSSM_PATH%" remove "%SERVICE_NAME%" confirm
if !errorlevel! equ 0 (
    echo [SUCCESS] Service '%SERVICE_NAME%' removed
) else (
    echo [ERROR] Failed to remove service
    exit /b 1
)
goto :eof

:: =============================================================
:status
:: =============================================================
echo [INFO] === Service Status: %SERVICE_NAME% ===
sc query "%SERVICE_NAME%" 2>nul
if !errorlevel! neq 0 (
    echo [WARNING] Service '%SERVICE_NAME%' is not installed
)
goto :eof

:: =============================================================
:main
:: =============================================================
call :check_admin
echo [INFO] === Metadata Editor FastAPI - Windows Service Installer (NSSM) ===
echo [INFO] Project directory: %PROJECT_DIR%

call :find_nssm
if !errorlevel! neq 0 exit /b 1

call :validate_inputs
if !errorlevel! neq 0 exit /b 1

call :create_directories
call :setup_shared_permissions
if !errorlevel! neq 0 exit /b 1

call :install_service
if !errorlevel! neq 0 exit /b 1

echo [SUCCESS] === Installation complete ===
endlocal
