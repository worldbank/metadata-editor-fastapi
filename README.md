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

## License

This project is licensed under the MIT License together with the [World Bank IGO Rider](WB-IGO-RIDER.md). The Rider is purely procedural: it reserves all privileges and immunities enjoyed by the World Bank, without adding restrictions to the MIT permissions. Please review both files before using, distributing or contributing.
