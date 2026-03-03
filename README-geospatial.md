# Geospatial Endpoints – Installation Guide

The geospatial features are **optional** and require additional dependencies that can be difficult to install in standard Python environments, particularly on Windows and some Linux distributions. The main challenge is **GDAL**, a native geospatial library that has complex system-level dependencies.

The recommended approach is to use **Miniconda3**, which provides pre-compiled binary packages for GDAL and related libraries via the `conda-forge` channel, avoiding the need to compile native extensions from source.

---

## Miniconda3

- Provides pre-built binaries for GDAL, Fiona, PROJ, and GEOS on all major platforms
- Avoids compiler toolchain requirements on Windows
- Isolates geospatial dependencies from the rest of your Python environment
- `conda-forge` channel ships up-to-date geospatial packages built consistently across platforms

---

## Step 1 – Install Miniconda3

Download and install Miniconda3 for your operating system from the official page:

**https://www.anaconda.com/docs/getting-started/miniconda/main**

### macOS (Apple Silicon or Intel)
```bash
# Download the installer (Apple Silicon)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Or for Intel Mac
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

# Run the installer
bash Miniconda3-latest-MacOSX-*.sh
```

### Linux (x86_64)
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### Windows
Download and run the `.exe` installer from the Miniconda download page. During installation, **check "Add Miniconda3 to my PATH environment variable"** — this is required for `start.bat` and `conda` commands to work from a standard Command Prompt or PowerShell window.

After installation, restart your terminal and verify:
```bash
conda --version
```

---

## Step 2 – Create a Conda Environment

Create a dedicated environment for this project using Python 3.11 (recommended for geospatial compatibility):

```bash
conda create -n metadata-editor python=3.11 -y
conda activate metadata-editor
```

---

## Step 3 – Install GDAL and Core Geospatial Libraries via conda-forge

Install the native geospatial libraries using conda before installing Python packages with pip. This ensures the correct compiled binaries are in place:

```bash
conda install -c conda-forge gdal fiona geopandas rasterio pyproj shapely -y
```

> **Note:** Do not install GDAL via pip in a conda environment — always install it through `conda-forge` to avoid binary compatibility issues.

---

## Step 4 – Install the Remaining Python Dependencies

With the conda environment active, install the core FastAPI dependencies and the remaining geospatial Python packages:

```bash
# Core application dependencies
pip install -r requirements.txt

# Remaining geospatial dependencies (GDAL/Fiona/GeoPandas are already installed via conda)
pip install metadataschemas pygeohash matplotlib
```

---

## Step 5 – Verify and Start the Application

To quickly verify the geospatial stack is working, you can run the application manually:

```bash
conda activate metadata-editor
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at `http://localhost:8000`. Check the `/docs` page and confirm geospatial endpoints are listed.

For normal use, stop the manual process and use the provided start/stop scripts instead — they manage the application as a background process and auto-detect the conda environment:

**macOS / Linux:**
```bash
./start.sh
./stop.sh
```

**Windows:**
```bat
start.bat
stop.bat
```

See [Using start.sh / start.bat with Conda](#using-startsh--startbat-with-conda) below for details.

> For production deployments on Linux, run the application as a managed systemd service instead.  
> → See the [Linux Service Installation Guide](deploy/linux/README.md)

---

## Using start.sh / start.bat with Conda

The project ships with `start.sh` (macOS/Linux) and `start.bat` (Windows) scripts that manage the application as a background process, including auto-detection of your Python environment.

### How environment detection works

Both scripts check for a Python environment in this order:

| Priority | Source | Condition |
|----------|--------|-----------|
| 1 | Conda env by name | A conda env named `metadata-editor` (or `$CONDA_ENV_NAME`) exists |
| 2 | Active conda env | `CONDA_DEFAULT_ENV` is set (you ran `conda activate` beforehand) |
| 3 | Virtual env `.venv` | `.venv/` directory exists in the project root |
| 4 | System Python | Any `python3.x` on PATH with uvicorn installed |

### macOS / Linux

If you created the conda environment in Step 2, the script detects it automatically — no manual activation required:

```bash
./start.sh        # auto-detects the 'metadata-editor' conda env
./stop.sh         # stop the background process
```

To use a different conda environment name:

```bash
CONDA_ENV_NAME=my-custom-env ./start.sh
```

### Windows

```bat
start.bat         :: auto-detects the 'metadata-editor' conda env
stop.bat          :: stop the background process
```

To use a different conda environment name:

```bat
set CONDA_ENV_NAME=my-custom-env
start.bat
```

> **Note:** On Windows, `start.bat` uses PowerShell's `Start-Process` to launch uvicorn in the background and capture its PID. PowerShell 5.1+ (included in Windows 10/11) is required.

### Verifying which environment was detected

Both scripts print the detected environment source at startup:

```
[INFO]   Environment: conda:metadata-editor
```

Possible values are `conda:<name>`, `conda-active:<name>`, `venv`, `system`, or `system-uvicorn`.

---

## Troubleshooting

### `ImportError: libgdal.so` or similar shared library errors (Linux)
Ensure you installed GDAL through conda-forge, not pip. Reinstall:
```bash
conda install -c conda-forge gdal --force-reinstall
```

### `ERROR 4: PROJ: ...` projection errors
Install or reinstall pyproj via conda-forge:
```bash
conda install -c conda-forge pyproj --force-reinstall
```

### Windows: DLL load failed
Make sure you used the conda-forge channel and that your conda environment is activated before running the application. Avoid mixing conda and pip installations for GDAL-related packages.

### Verifying the installation
You can verify the geospatial stack is correctly installed by running:
```bash
python -c "from osgeo import gdal; import fiona; import geopandas; import rasterio; print('Geospatial stack OK')"
```

---

## Environment Summary

| Package     | Install via     | Notes                                      |
|-------------|-----------------|---------------------------------------------|
| gdal        | conda-forge     | Must be installed via conda, not pip        |
| fiona       | conda-forge     | Depends on GDAL                             |
| geopandas   | conda-forge     | Depends on Fiona, Shapely, PyProj           |
| rasterio    | conda-forge     | Depends on GDAL                             |
| pyproj      | conda-forge     | Depends on PROJ native library              |
| shapely     | conda-forge     | Geometry engine                             |
| matplotlib  | pip             | Visualization                               |
| metadataschemas | pip         | Metadata schema support                     |
| pygeohash   | pip             | Geohash utilities                           |

---

## Deactivating and Removing the Environment

```bash
# Deactivate the current environment
conda deactivate

# Remove the environment entirely (if needed)
conda env remove -n metadata-editor
```
