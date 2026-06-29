# Linux Service Installation – editor-fastapi

This guide covers running the Metadata Editor FastAPI backend as a managed **systemd service** on Linux.

---

## What this service is

The FastAPI backend is a **local processing worker** for the [Metadata Editor](https://github.com/worldbank/metadata-editor) web application. It is **not** a standalone public API.

It handles CPU-intensive tasks the PHP editor delegates to Python: reading Stata/SPSS/CSV files, generating data dictionaries, geospatial metadata extraction, and similar operations. The editor passes **absolute file paths** on shared storage; this service reads and writes those files on the same machine.

### Typical architecture

```
┌─────────────────────────────────────────────────────────┐
│  Same Linux host                                        │
│                                                         │
│  Browser ──► Apache/NGINX + PHP (Metadata Editor)       │
│                    │                                    │
│                    │  http://127.0.0.1:8000             │
│                    ▼                                    │
│              editor-fastapi (this service)              │
│                    │                                    │
│                    ▼                                    │
│         /var/www/metadata-editor/datafiles/  ◄── shared │
└─────────────────────────────────────────────────────────┘
```

Both the web server and FastAPI need read/write access to the editor data folder. The install script can configure shared group permissions for that (see [How shared storage access works](#how-shared-storage-access-works)).

---

## Security model

**This service must not be exposed to the internet or untrusted networks.**

| Control | What to do |
|---------|------------|
| **Bind address** | Default is `127.0.0.1` — only local processes can connect. The Metadata Editor on the same host calls `http://127.0.0.1:8000`. |
| **`STORAGE_PATH` in `.env`** | **Required explicit choice.** Set an absolute path to the editor data folder, or `STORAGE_PATH=` (empty) to disable path validation — empty is for local development only. |
| **Firewall** | Do not forward port 8000 publicly. If you must bind to `0.0.0.0` on an internal network, restrict access with firewall rules. |
| **Service user** | Run as dedicated user `editor-fastapi` (created by the install script), not root. |
| **Authentication** | Not built in — security relies on localhost binding and network isolation. |

The install script defaults to `HOST=127.0.0.1`. Override with `HOST=0.0.0.0` only when the editor and API run on the same host but you need a non-loopback bind (unusual).

---

## Prerequisites

- Linux with systemd (Ubuntu 18.04+, Debian 10+, RHEL/CentOS/Rocky 8+)
- Miniconda3 installed with the `metadata-editor` conda environment set up  
  → See [README-geospatial.md](../../README-geospatial.md) for conda environment setup
- The application deployed to its target directory (e.g. `/opt/metadata-editor` or `/var/www/metadata-editor`)
- A `.env` file in the project root — copy from `.env.example` and configure at minimum:

```bash
cp .env.example .env
```

**Production `.env` example:**

```bash
STORAGE_PATH=/var/www/metadata-editor/datafiles
HOST=127.0.0.1
PORT=8000
```

- Apache or NGINX with PHP running (Metadata Editor), on the **same host** as this service

---

## Two different `STORAGE_PATH` settings

| Where | Required? | Purpose |
|-------|-----------|---------|
| **`.env` file** (runtime) | **Yes — explicit choice** | Application path sandboxing. Set a directory path, or empty to disable validation (dev only). |
| **Install script variable** (one-time) | No | Optional — only runs `chown`/`chmod` on the shared data folder during install |

You can set both to the same path for a typical install. Skipping the install-script variable only means you configure filesystem permissions manually afterward.

---

## Files in this directory

| File | Purpose |
|------|---------|
| `editor-fastapi.service` | systemd unit template — do not edit directly |
| `install-service.sh` | Installs, configures, and starts the service |

---

## Step 1 – Find your Python path

The install script requires an explicit path to the Python executable inside the conda environment. Run this as the user who owns the conda installation (not root):

```bash
conda activate metadata-editor
which python
```

Example output:
```
/home/myuser/miniconda3/envs/metadata-editor/bin/python
```

Copy this path — you will need it in Step 3.

---

## Step 2 – Identify your web server user

The shared storage folder must be accessible to both the FastAPI service and the PHP web application. You need to know the OS user your web server runs as:

| Web server | Typical user | How to verify |
|------------|-------------|---------------|
| Apache on Ubuntu/Debian | `www-data` | `ps aux \| grep apache` |
| Apache on RHEL/Rocky/CentOS | `apache` | `ps aux \| grep httpd` |
| NGINX | `www-data` or `nginx` | `ps aux \| grep nginx` |
| PHP-FPM pool | custom user | check `/etc/php-fpm.d/*.conf` |

---

## Step 3 – Run the install script

The script must be run as root. The only required variable is `CONDA_PYTHON_PATH`.

Ensure `.env` exists with `STORAGE_PATH` set before starting the service (see [Prerequisites](#prerequisites)).

### Minimal install (no shared storage permission setup)

```bash
sudo CONDA_PYTHON_PATH=/home/myuser/miniconda3/envs/metadata-editor/bin/python \
     bash deploy/linux/install-service.sh
```

### Full install with shared storage permissions

```bash
sudo \
  CONDA_PYTHON_PATH=/home/myuser/miniconda3/envs/metadata-editor/bin/python \
  STORAGE_PATH=/var/www/metadata-editor/datafiles \
  WEB_SERVER_USER=www-data \
  bash deploy/linux/install-service.sh
```

### All available variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CONDA_PYTHON_PATH` | **Yes** | — | Absolute path to `python` in the conda env |
| `PROJECT_DIR` | No | parent of `deploy/` | Application root (where `main.py` lives) |
| `SERVICE_USER` | No | `editor-fastapi` | OS user the service runs as |
| `HOST` | No | `127.0.0.1` | Bind address — keep as localhost unless you have a specific reason to change |
| `PORT` | No | `8000` | Listening port |
| `SHARED_GROUP` | No | `editor-shared` | Group shared between FastAPI and web server |
| `WEB_SERVER_USER` | No | `www-data` | Web server OS user added to shared group |
| `STORAGE_PATH` | No | *(unset)* | Install-time only: path for `chown`/`chmod` on shared storage |

---

## How shared storage access works

When FastAPI and a PHP application both need read/write access to the same folder, a **shared group** is used. Neither service needs to run as the other's user.

### What the install script does

1. Creates a shared OS group (`editor-shared` by default)
2. Adds the FastAPI service user (`editor-fastapi`) to the group
3. Adds the web server user (`www-data`) to the group
4. If install `STORAGE_PATH` is set:
   - Sets ownership: `chown -R www-data:editor-shared /path/to/storage`
   - Sets permissions: `chmod -R 2775 /path/to/storage`

### Why `2775` (setgid)?

```
2  = setgid bit — new files and subdirectories inherit the group automatically
7  = owner: read + write + execute
7  = group: read + write + execute
5  = others: read + execute
```

Without the setgid bit, files created by the FastAPI process would inherit the `editor-fastapi` primary group, making them inaccessible to `www-data`. The setgid bit eliminates the need for any post-write `chmod` calls.

### Why `UMask=0002` in the service?

The service unit sets `UMask=0002`, which means every file created by the FastAPI process is group-writable by default (`rw-rw-r--`). Combined with the setgid directory, both the PHP application and FastAPI can modify each other's files.

### Diagram

```
/var/www/metadata-editor/datafiles/
  owner : www-data
  group : editor-shared       ← both users are members of this group
  mode  : 2775 (setgid)

  www-data       → member of editor-shared → full read/write ✓
  editor-fastapi → member of editor-shared → full read/write ✓
```

### If you skipped install `STORAGE_PATH`

Set permissions manually after the service is running:

```bash
sudo chown -R www-data:editor-shared /path/to/storage
sudo chmod -R 2775 /path/to/storage
```

### RHEL / CentOS / Rocky — different web server user

If your web server runs as `apache` instead of `www-data`:

```bash
sudo \
  CONDA_PYTHON_PATH=/path/to/python \
  WEB_SERVER_USER=apache \
  STORAGE_PATH=/path/to/storage \
  bash deploy/linux/install-service.sh
```

---

## Managing the service

```bash
# Check status
systemctl status editor-fastapi

# Start / stop / restart
systemctl start editor-fastapi
systemctl stop editor-fastapi
systemctl restart editor-fastapi

# View live logs
journalctl -u editor-fastapi -f

# View last 100 log lines
journalctl -u editor-fastapi -n 100 --no-pager

# Enable / disable auto-start at boot
systemctl enable editor-fastapi
systemctl disable editor-fastapi
```

---

## Uninstalling

```bash
sudo bash deploy/linux/install-service.sh --uninstall
```

This stops the service, disables it from starting at boot, and removes the unit file from `/etc/systemd/system/`. It does not remove the `editor-fastapi` OS user, the `editor-shared` group, or the application files.

To remove the service user and group manually:

```bash
sudo userdel editor-fastapi
sudo groupdel editor-shared
```

---

## Troubleshooting

### Service fails to start

Check the logs immediately after the failure:

```bash
journalctl -u editor-fastapi -n 50 --no-pager
```

Common causes:

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `STORAGE_PATH must be set explicitly` | Missing from `.env` | Copy `.env.example` to `.env` and set `STORAGE_PATH` |
| `STORAGE_PATH does not exist` | Wrong path in `.env` | Create the directory or fix the path |
| `No such file or directory` on `ExecStart` | Wrong `CONDA_PYTHON_PATH` | Verify path with `ls -la /path/to/python` |
| `ModuleNotFoundError: uvicorn` | Wrong Python env | Re-run with correct `CONDA_PYTHON_PATH` |
| `ImportError: libgdal.so` | Missing `LD_LIBRARY_PATH` | Check `CONDA_LIB_PATH` in the installed unit file |
| `Permission denied` on `.env` or `main.py` | Service user can't read files | `chown -R editor-fastapi:editor-fastapi /opt/metadata-editor` |
| `Address already in use` | Another process on port 8000 | Change `PORT` or stop the conflicting process |

### Inspect the generated unit file

```bash
cat /etc/systemd/system/editor-fastapi.service
```

Verify all placeholder tokens (`%%...%%`) were replaced correctly — any remaining token means a variable was not set during install. Confirm `--host 127.0.0.1` unless you intentionally overrode `HOST`.

### Verify shared group membership

```bash
getent group editor-shared
# Expected output: editor-shared:x:1234:editor-fastapi,www-data
```

### Test file permissions manually

```bash
# As the FastAPI service user
sudo -u editor-fastapi touch /path/to/storage/test-fastapi.txt
ls -la /path/to/storage/test-fastapi.txt
# group should be 'editor-shared' and mode should include g+w

# As the web server user
sudo -u www-data cat /path/to/storage/test-fastapi.txt
sudo -u www-data rm /path/to/storage/test-fastapi.txt
```

### Verify localhost binding

```bash
ss -tlnp | grep 8000
# Should show 127.0.0.1:8000, not 0.0.0.0:8000 (unless you changed HOST)
```

---

## See also

- [Main README](../../README.md) — security model, `.env` configuration, start scripts
- [Windows service install](../windows/README.md) — equivalent guide for IIS/Windows Server
