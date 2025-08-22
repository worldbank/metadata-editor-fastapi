# Metadata Editor – FastAPI Backend Service

A FastAPI-based RESTful service that processes data files (Stata, SPSS, CSV) to support the [Metadata Editor](https://github.com/worldbank/metadata-editor).

## ✨ Features

- 🔍 Generate **data dictionaries** compatible with **DDI CodeBook 2.5**
- 📊 Produce **summary statistics** and **frequencies**
- 🔄 Support for **importing and exporting** data in:
  - **SPSS (.sav)**
  - **Stata (.dta)**
  - **CSV (.csv)**

## 🔗 Integration

This service is designed to be used in conjunction with the [Metadata Editor web application](https://github.com/worldbank/metadata-editor), enhancing its ability to automate data processing and metadata generation workflows.


## Requirements
Python 3.9 or later

## Dependencies

```
fastapi==0.109.0
numpy==1.26.3
pandas==2.1.4
pydantic==1.10.7
pyreadstat==1.2.6
statsmodels==0.14.1
uvicorn==0.17.*
```

## Installation

### Option 1: Direct Installation
```
pip install -r requirements.txt
```

### Option 2: Using Virtual Environment (Recommended)
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# To deactivate the virtual environment when done
deactivate
```

## Start web app

### If using Option 1 (Direct Installation):
```bash
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### If using Option 2 (Virtual Environment):
```bash
# Make sure the virtual environment is activated
source venv/bin/activate

# Start the application
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# When done, deactivate the virtual environment
deactivate
```

The application will be available at `http://localhost:8000`

## Configuration

### Storage Path Configuration
The `STORAGE_PATH` should point to the folder used by the metadata editor for data storage. 

```bash
# Set the path to your data files directory
STORAGE_PATH=/path/to/your/metadata-editor/datafiles
```

**Important Notes:**
- **Permissions**: The application must have read/write access to this directory
- **Path Format**: Use absolute paths (e.g., `/Volumes/webdev/data` on macOS/Linux, `C:\data` on Windows)
- **Validation**: The application will fail to start if the path doesn't exist or is inaccessible


### Logging Configuration

To enable error logging, copy the contents of `env_configuration_example.txt` to end of the `.env` file and modify as needed:

Control logging verbosity and output through environment variables:

```bash
# Production (clean output, errors only)
LOG_LEVEL=ERROR
LOG_FORMAT=simple
LOG_TO_FILE=false

# Development (detailed debugging)
LOG_LEVEL=DEBUG
LOG_FORMAT=detailed
LOG_TO_FILE=false

# File logging with timestamps
LOG_LEVEL=INFO
LOG_FORMAT=timestamp
LOG_TO_FILE=true
# Log file path (only used if LOG_TO_FILE=true)
# Default: logs/error-YYYY-MM-DD.log
LOG_FILE_PATH=logs/error-2025-08-22.log
```

### Complete Setup Example
Here's a complete `.env` file setup for a typical development environment:

```bash
# Storage configuration
STORAGE_PATH=/Users/username/projects/metadata-editor/datafiles

# Logging configuration (development mode)
LOG_LEVEL=DEBUG
LOG_FORMAT=detailed
LOG_TO_FILE=false

# Job management
CLEANUP_INTERVAL_HOURS=1
MAX_JOB_AGE_HOURS=24
MAX_MEMORY_JOBS=500

# Server configuration
HOST=127.0.0.1
PORT=8000
RELOAD=true
```

## License

This project is licensed under the MIT License together with the [World Bank IGO Rider](WB-IGO-RIDER.md). The Rider is purely procedural: it reserves all privileges and immunities enjoyed by the World Bank, without adding restrictions to the MIT permissions. Please review both files before using, distributing or contributing.
