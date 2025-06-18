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

```
pip install -r requirements.txt
```

## Start web app

```
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## License

This project is licensed under the MIT License together with the [World Bank IGO Rider](WB-IGO-RIDER.md). The Rider is purely procedural: it reserves all privileges and immunities enjoyed by the World Bank, without adding restrictions to the MIT permissions. Please review both files before using, distributing or contributing.
