# Running editor-fastapi as a Windows Service (NSSM)

This guide walks through installing the Metadata Editor FastAPI backend as a
Windows service using **NSSM** (Non-Sucking Service Manager).

NSSM wraps any executable as a proper Windows service, handles restarts, and
routes stdout/stderr to log files automatically.

---

## Prerequisites

| Requirement | How to verify |
|---|---|
| Windows 10 / Server 2016 or later | `winver` |
| Conda environment configured per [README-geospatial.md](../../README-geospatial.md) | `conda activate metadata-editor` |
| uvicorn installed | `python -m uvicorn --version` |
| Administrator rights | Run CMD as Administrator |

---

## Step 1 — Install NSSM

**Option A — Chocolatey (recommended)**

```bat
choco install nssm
```

**Option B — Manual download**

1. Download the latest zip from <https://nssm.cc/download>
2. Extract it — e.g. to `C:\tools\nssm\`
3. Add the `win64` folder to your system PATH, **or** set `NSSM_PATH` when
   running the install script:

```bat
set NSSM_PATH=C:\tools\nssm\win64\nssm.exe
```

---

## Step 2 — Find the Conda Python Path

The install script requires the **exact path** to `python.exe` inside the conda
environment. Do not let the script guess it.

Open a normal (non-admin) Command Prompt and run:

```bat
conda activate metadata-editor
where python
```

The output will look like one of:

```
C:\Users\myuser\miniconda3\envs\metadata-editor\python.exe
C:\ProgramData\miniconda3\envs\metadata-editor\python.exe
```

Copy this path — you will need it in the next step.

---

## Step 3 — Install the Service

Open an **Administrator** Command Prompt and set variables, then run the script.

### Minimal install

```bat
set CONDA_PYTHON_PATH=C:\Users\myuser\miniconda3\envs\metadata-editor\python.exe
cd /d C:\path\to\metadata-editor-fastapi\deploy\windows
install-service.bat
```

### Full install (with shared storage permissions)

```bat
set CONDA_PYTHON_PATH=C:\Users\myuser\miniconda3\envs\metadata-editor\python.exe
set STORAGE_PATH=C:\inetpub\wwwroot\metadata-editor\datafiles
set WEB_SERVER_USER=IIS_IUSRS
install-service.bat
```

### All available variables

| Variable | Default | Description |
|---|---|---|
| `CONDA_PYTHON_PATH` | **(required)** | Absolute path to `python.exe` in the conda env |
| `PROJECT_DIR` | Two levels up from `deploy\windows\` | Application root where `main.py` lives |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Listening port |
| `NSSM_PATH` | Searched on `%PATH%` | Full path to `nssm.exe` |
| `SHARED_GROUP` | `editor-shared` | Local group for shared storage access |
| `WEB_SERVER_USER` | `IIS_IUSRS` | Web server account added to the shared group |
| `STORAGE_PATH` | *(not set)* | Folder to apply shared permissions to (optional) |

---

## Shared Storage Permissions

When the FastAPI backend and a web server (IIS / Apache) both need read/write
access to the same folder (e.g. uploaded files), the install script handles
this automatically when `STORAGE_PATH` is set.

### What the script does

```
Local group: editor-shared
    ├── IIS_IUSRS   (web server – reads/downloads files)
    └── NETWORK SERVICE / LOCAL SERVICE (service account – writes files)

Storage folder
    └── icacls grant editor-shared:(OI)(CI)F   ← full control, inherited
    └── icacls grant NETWORK SERVICE:(OI)(CI)F
    └── icacls grant LOCAL SERVICE:(OI)(CI)F
```

`(OI)(CI)` means the ACE propagates to all files and sub-folders created
later — equivalent to the Linux setgid directory pattern.

### Web server user reference

| Web server | Default account |
|---|---|
| IIS application pool (default) | `IIS_IUSRS` |
| IIS application pool (Classic) | `IUSR` |
| Apache (XAMPP) | `SYSTEM` or the user running Apache |
| Apache (custom service) | Check the service's **Log On** tab in Services |

To verify the account your IIS app pool uses:

1. Open **IIS Manager**
2. Select the application pool → **Advanced Settings**
3. Look at **Identity**

---

## Service Management

After installation, manage the service from an Administrator Command Prompt:

```bat
:: Start / stop / restart
sc start  editor-fastapi
sc stop   editor-fastapi
sc query  editor-fastapi

:: View real-time status
nssm status editor-fastapi

:: Edit service settings interactively (GUI)
nssm edit editor-fastapi
```

Or use Windows Services (`services.msc`) — look for **editor-fastapi**.

---

## Log Files

Logs are written to the `logs\` folder inside the application directory:

| File | Contents |
|---|---|
| `logs\service.log` | Combined stdout — application startup, requests |
| `logs\service_err.log` | Stderr — Python exceptions, tracebacks |

Logs rotate automatically when they reach 10 MB.

To tail logs in PowerShell:

```powershell
Get-Content "C:\path\to\metadata-editor-fastapi\logs\service.log" -Wait -Tail 50
```

---

## Uninstall

```bat
install-service.bat --uninstall
```

This stops and removes the service. Application files and log files are not
removed.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `This script must be run as Administrator` | CMD not elevated | Right-click → Run as administrator |
| `NSSM not found` | NSSM not installed or not on PATH | Install NSSM or set `NSSM_PATH` (see Step 1) |
| `CONDA_PYTHON_PATH is not set` | Variable not set | Run `where python` inside the conda env (see Step 2) |
| `CONDA_PYTHON_PATH does not exist` | Wrong path copied | Re-run `where python` exactly as shown |
| `uvicorn is not installed` | Wrong conda env activated | Activate `metadata-editor` env before running `where python` |
| `main.py not found` | `PROJECT_DIR` wrong | Set `PROJECT_DIR` to the folder containing `main.py` |
| Service fails to start | GDAL DLLs not found | Check `service_err.log`; ensure `CONDA_LIB_DIR` in PATH (auto-set by script) |
| `500 errors` on geospatial routes | Missing GDAL env | Verify with `nssm edit editor-fastapi` → **Environment** tab |
| Port already in use | Another process on port | Change `PORT` or stop the conflicting process: `netstat -ano \| findstr :8000` |
| Log file access denied | Not running as Administrator | UAC elevation required for `C:\Windows\` paths; keep logs inside project dir |
| Service not visible in services.msc | Install failed silently | Check Command Prompt output; re-run with Administrator CMD |

### Inspecting service failures with Event Viewer

1. Open **Event Viewer** (`eventvwr.msc`)
2. Navigate to **Windows Logs → Application**
3. Filter by source **NSSM** or **editor-fastapi**

---

## See Also

- [Linux service setup (systemd)](../linux/README.md)
- [Geospatial environment setup (Miniconda3)](../../README-geospatial.md)
- [NSSM documentation](https://nssm.cc/usage)
