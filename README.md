# Metadata Editor – FastAPI Backend Service

A FastAPI-based RESTful service that processes data files (Stata, SPSS, CSV) to support the [Metadata Editor](https://github.com/worldbank/metadata-editor). It is designed to run on the same machine as the Metadata Editor and is not intended for public internet exposure.

## ✨ Features

- Generate **data dictionaries** compatible with **DDI CodeBook 2.5**
- Produce **summary statistics** and **frequencies**
- Support for **importing and exporting** data in:
  - **SPSS (.sav)**
  - **Stata (.dta)**
  - **CSV (.csv)**
- **Geospatial metadata endpoints** (optional – see [Geospatial Installation Guide](README-geospatial.md))
- **AI metadata reviewer** (optional – see [Metadata Reviewer Installation Guide](README-reviewer.md))

## Integration

This service is designed to be used in conjunction with the [Metadata Editor web application](https://github.com/worldbank/metadata-editor), enhancing its ability to automate data processing and metadata generation workflows.

## Security model

This FastAPI service is a **local processing worker**, not a public API. It reads and writes files on paths supplied by the Metadata Editor (Stata/SPSS/CSV conversion, data dictionaries, geospatial metadata, and similar tasks).

**Do not expose this service to untrusted networks.** Run it on the same machine as the Metadata Editor and bind to localhost (`127.0.0.1`).

| Control | Recommendation |
|---------|----------------|
| Network | `HOST=127.0.0.1` (default). Use `0.0.0.0` only on trusted internal networks with a firewall. |
| `STORAGE_PATH` | **Required in `.env`** — set an absolute directory path, or `STORAGE_PATH=` (empty) to disable path validation for local development only. |
| OS permissions | Run under a dedicated service account with access limited to editor data folders. |

Copy `.env.example` to `.env` before starting. The application **will not start** unless `STORAGE_PATH` is explicitly set in `.env`.

**Production deployment (recommended on a server):**

- Linux (systemd): [deploy/linux/README.md](deploy/linux/README.md)
- Windows (NSSM service): [deploy/windows/README.md](deploy/windows/README.md)

## Requirements

- Python **3.11+**
- [Miniconda3](https://www.anaconda.com/docs/getting-started/miniconda/main) — recommended for production and geospatial features
- Metadata Editor web app on the **same machine**

Core Python dependencies are listed in [`requirements.txt`](requirements.txt).

---

## Installation

### 1. Configure environment (required)

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

- `STORAGE_PATH` — absolute path to the Metadata Editor data folder (see [Configuration](#storage-path-storage_path) below)
- `HOST=127.0.0.1` — already the default in `.env.example`

### 2. Python environment

Use **one** of the options below. Choose **Option 1 (Conda)** when you need geospatial endpoints or are on Windows. Choose **Option 2 (`.venv`)** when you only need core features — suitable for development and production alike.

#### Option 1: Conda (`metadata-editor` env) — recommended

Best when you need **geospatial** endpoints or are installing on **Windows**. GDAL and related native libraries come from `conda-forge`.

Full step-by-step: **[Geospatial Installation Guide](README-geospatial.md)** — follow Steps 1–4 even if you only need core features today.

Summary:

```bash
conda create -n metadata-editor python=3.11 -y
conda activate metadata-editor
conda install -c conda-forge gdal fiona geopandas rasterio pyproj shapely -y   # skip if core-only
pip install -r requirements.txt
pip install metadataschemas pygeohash matplotlib   # geospatial only; see README-geospatial.md
```

On Windows, enable “Add Miniconda3 to PATH” during install so `start.bat` and the Windows service installer can find conda.

#### Option 2: Virtual environment (core features, no geospatial)

Lightweight option when you do **not** need geospatial endpoints — works well for local development and for production when you only need Stata, SPSS, and CSV processing. Use **`.venv`**; the start scripts look for this directory name.

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows:**

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

#### Option 3: System Python (not recommended)

Only if you cannot use Conda or a virtual environment — for example, a throwaway test machine or a container where `python` is already isolated.

```bash
pip install -r requirements.txt
```

> **Warning:** This installs into the active Python environment (often system-wide). It can conflict with OS-managed packages on Linux, does not support geospatial endpoints, and is unsuitable for production. Prefer Option 1 or Option 2.

### Optional: Geospatial endpoints

Requires the Conda + `conda-forge` setup in Option 1. See **[README-geospatial.md](README-geospatial.md)**.

### Optional: Metadata reviewer

```bash
pip install -r requirements-reviewer.txt
cp reviewer.env.example reviewer.env
# edit reviewer.env with your LLM provider credentials
```

See **[README-reviewer.md](README-reviewer.md)**.

---

## Running the application

### Development (recommended): start/stop scripts

Scripts auto-detect Python in this order: conda env `metadata-editor` → active conda env → `.venv` → system Python. They read `HOST` and `PORT` from the environment / `.env` and default to **`127.0.0.1:8000`**.

**Linux / macOS:**

```bash
./start.sh -f      # foreground — best for debugging (Ctrl+C to stop)
./start.sh         # background
./stop.sh          # stop gracefully
./start.sh --help  # all options
```

**Windows:**

```bat
start.bat -f       :: foreground
start.bat          :: background
stop.bat           :: stop gracefully
start.bat --help
```

The API is at `http://127.0.0.1:8000` (localhost only by default).

### Manual start (debugging only)

With your conda env, `.venv`, or system Python configured and `.env` in place:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Add `--reload` for auto-reload during development (not used by the start scripts or production services).

### Production: OS service

Do **not** rely on `start.sh` / `start.bat` for a production Metadata Editor server. Install as a service so the API starts at boot and runs under a dedicated account:

| OS | Guide | Mechanism |
|----|--------|-----------|
| **Linux** | [deploy/linux/README.md](deploy/linux/README.md) | systemd (`install-service.sh`) |
| **Windows** | [deploy/windows/README.md](deploy/windows/README.md) | NSSM (`install-service.bat`) |

Both guides install via an explicit Python path (commonly Conda `metadata-editor`). A `.venv` interpreter works the same way when geospatial packages are not required.

---

## Configuration

### Environment file

```bash
cp .env.example .env
```

### Storage path (`STORAGE_PATH`)

`STORAGE_PATH` must be **explicitly set** in `.env`. The application will refuse to start if it is missing.

| Value | Behavior |
|-------|----------|
| Absolute directory path | Restricts file operations on endpoints that read or write user-supplied paths |
| Empty (`STORAGE_PATH=`) | Disables path validation — **local development only** |

```bash
# Production — point at the metadata editor data folder
STORAGE_PATH=/path/to/your/metadata-editor/datafiles

# Local development only — disable path validation
# STORAGE_PATH=
```

**Notes:**
- Use **absolute paths** (e.g. `/var/www/metadata-editor/datafiles` on Linux, `C:\inetpub\metadata-editor\datafiles` on Windows)
- The directory must exist when a path is set; the application validates this at startup
- The service account must have read/write access to this directory

### Logging configuration

Copy variables from `logging_config_example.env` into your `.env` file and adjust as needed.

Logs include a timestamp on each line. By default, output goes to the console and to `logs/app.log`, with a new file created at midnight (`logs/app.log.YYYY-MM-DD`). Log files are appended across restarts and retained for 30 days.

```bash
# Production (errors only, file + console)
LOG_LEVEL=ERROR
LOG_FORMAT=timestamp
LOG_TO_FILE=true

# Development (detailed debugging)
LOG_LEVEL=DEBUG
LOG_FORMAT=detailed
LOG_TO_FILE=true

# Console only (no file)
LOG_LEVEL=INFO
LOG_FORMAT=timestamp
LOG_TO_FILE=false

# Optional overrides
# LOG_FILE_PATH=logs/app.log
# LOG_RETENTION_DAYS=30
```

### Complete setup example

Example `.env` for local development:

```bash
# Storage — required (use empty STORAGE_PATH= for dev-only validation off)
STORAGE_PATH=/path/to/metadata-editor/datafiles

# Server
HOST=127.0.0.1
PORT=8000

# Logging
LOG_LEVEL=DEBUG
LOG_FORMAT=detailed
LOG_TO_FILE=true

# Job management
CLEANUP_INTERVAL_HOURS=1
MAX_JOB_AGE_HOURS=24
MAX_MEMORY_JOBS=500
```

## License

This project is licensed under the MIT License together with the [World Bank IGO Rider](WB-IGO-RIDER.md). The Rider is purely procedural: it reserves all privileges and immunities enjoyed by the World Bank, without adding restrictions to the MIT permissions. Please review both files before using, distributing or contributing.
