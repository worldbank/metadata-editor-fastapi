# FastAPI Backend service for Metadata Editor data management
A REST API for processing data files (Stata, SPSS, CSV) 

Features:
- Generate data dictionaries compatible with DDI CodeBook 2.5
- Generate summary statistics and frequencies 
- Export data from SPSS, STATA formats to CSV


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
