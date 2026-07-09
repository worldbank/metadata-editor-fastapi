#import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pyreadstat
import time
from typing import List, Optional, Dict, Any
from src.DataUtils import DataUtils
from src.DataDictionary import DataDictionary
from src.DataDictionaryCsv import DataDictionaryCsv
from src.ExportDatafile import ExportDatafile
from src.routers.geospatial import router as geospatial_router
from src.routers.timeseries import router as timeseries_router
from src.reviewer import dispose_reviewer_job_if_needed, register_reviewer
from src.version import get_version
import re
import pandas as pd
import numpy as np
import math
import os
from pathlib import Path
#from pydantic import BaseSettings
from pydantic_settings import BaseSettings
import json
from src.DictParams import DictParams
from src.utils.dta_reader import read_dta, write_dta_to_csv
import asyncio
import functools
import hashlib
import datetime
from fastapi.concurrency import run_in_threadpool
import shutil
import glob
from dotenv import load_dotenv
import traceback
import logging

from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from starlette.exceptions import HTTPException as StarletteHTTPException
from src.logging_config import install_asyncio_exception_handler, setup_logging
from src.job_queue import enqueue_fifo_job, recover_pending_jobs
from src.job_store import JobStore

# Load environment variables from files next to this module (stable regardless of cwd)
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

# Setup logging with configuration (after loading environment variables)
logger = setup_logging(_PROJECT_ROOT)

# Cleanup configuration
# run cleanup task to remove old jobs
CLEANUP_INTERVAL_HOURS = int(os.getenv("CLEANUP_INTERVAL_HOURS", "1"))
# remove jobs older than this
MAX_JOB_AGE_HOURS = int(os.getenv("MAX_JOB_AGE_HOURS", "24"))

# limit the number of jobs in memory
MAX_MEMORY_JOBS = int(os.getenv("MAX_MEMORY_JOBS", "500"))

from src.utils.path_security import (
    resolve_safe_path,
    resolve_safe_paths,
    resolve_safe_path_http,
    resolve_safe_paths_http,
)

#class Settings(BaseSettings):
#    storage_path: str = "data"    
    

#settings = Settings()


class FileInfo(BaseModel):
    file_path: str


class NameLabelsParams(BaseModel):
    """Parameters for /name-labels (backward compatible with FileInfo)."""
    file_path: str
    expected_columns: Optional[List[str]] = None
    include_file_info: bool = False
    include_comparison: bool = False
    columns_only: bool = False


class WeightsColumns(BaseModel):
    weight_field: str
    field: str

class UserMissings(BaseModel):
    field: str
    missings: List[str]
    
class VarInfo(BaseModel):
    file_path: str
    var_names: List[str]
    weights: List[WeightsColumns] = []
    missings: List[UserMissings] = []

class DataProcessingParams(BaseModel):
    """Parameters for unified data processing (CSV generation + data dictionary)"""
    file_path: str
    generate_csv: bool = True  # Whether to generate CSV
    generate_data_dictionary: bool = True  # Whether to generate data dictionary
    
    # Data dictionary specific parameters (optional)
    var_names: List = []
    weights: List[WeightsColumns] = []
    missings: Optional[Dict[str, Any]] = {}
    dtypes: Dict[str, Any] = {}
    value_labels: Dict[str, Any] = {}
    name_labels: Dict[str, Any] = {}
    categorical: List[str] = []
    export_format: str = "csv"


class RemoveColumnsParams(BaseModel):
    """Parameters for removing columns from a CSV file. Writes to a new file; caller replaces original if desired."""
    file_path: str
    column_names: List[str]
    output_path: str


datadict=DataDictionary()

app = FastAPI()
app.fifo_queue = asyncio.Queue()

app.jobs = {}
app.job_store = JobStore()

# Include geospatial router
app.include_router(geospatial_router)
# Include timeseries router
app.include_router(timeseries_router)
register_reviewer(app, _PROJECT_ROOT)

# Cleanup metrics
class CleanupMetrics:
    def __init__(self):
        self.last_cleanup = None
        self.jobs_cleaned_total = 0
        self.files_removed_total = 0
        self.cleanup_duration_seconds = 0

cleanup_metrics = CleanupMetrics()


@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request, exc):
    if exc.status_code >= 500:
        logger.error(
            "HTTP %s %s: %s",
            request.method,
            request.url.path,
            exc.detail,
            exc_info=True,
        )
    else:
        logger.debug(
            "HTTP %s %s: %s",
            request.method,
            request.url.path,
            exc.detail,
        )
    return await http_exception_handler(request, exc)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})



@app.get("/")
async def root(request: Request):
    return {
        "message": "PyDataTool API - See documentation at " + str(request.url) + "docs",
        "version": get_version()
    }

@app.get("/status")
async def status():
    return {"status": "ok"}

@app.get("/version")
async def version():
    """Get application version."""
    return {"version": get_version()}


@app.post("/metadata")
async def metadata(fileinfo: FileInfo):
    file_path = resolve_safe_path_http(fileinfo.file_path, label="file_path")
    datadict=DataDictionary()
    return datadict.get_metadata(fileinfo.model_copy(update={"file_path": file_path}))

@app.post("/name-labels")
async def name_labels(params: NameLabelsParams):
    """
    Return variable names/labels from a data file.

    Optional flags (backward compatible — omit for previous behavior):
    - include_file_info: add file_info (format, format_version, format_label)
    - expected_columns + include_comparison: compare columns to expected names
    - columns_only: return column_names only (skip per-variable labels)
    """
    file_path = resolve_safe_path_http(params.file_path, label="file_path")
    file_ext = os.path.splitext(file_path)[1].lower()

    # CSV: header-only inspect (DataDictionary is Stata/SPSS)
    if file_ext == ".csv":
        from src.utils.source_file_info import build_file_info, compare_columns

        encodings_to_try = [None, "utf-8", "latin1", "cp1252", "iso-8859-1"]
        last_error = None
        df = None
        for encoding in encodings_to_try:
            try:
                if encoding is None:
                    df = pd.read_csv(file_path, nrows=0)
                else:
                    df = pd.read_csv(file_path, nrows=0, encoding=encoding)
                break
            except Exception as e:
                last_error = e
                continue
        if df is None:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read CSV header: {last_error}",
            )

        # Row count without loading full file
        row_count = 0
        try:
            with open(file_path, "rb") as fh:
                row_count = max(sum(1 for _ in fh) - 1, 0)
        except OSError:
            row_count = None

        column_names = list(df.columns)
        if params.columns_only:
            result = {
                "rows": row_count,
                "columns": len(column_names),
                "column_names": column_names,
                "variables": [],
            }
        else:
            result = {
                "rows": row_count,
                "columns": len(column_names),
                "variables": [
                    {"name": name, "labl": None, "var_format": None}
                    for name in column_names
                ],
            }
        if params.include_file_info:
            result["file_info"] = build_file_info(file_path)
        if params.include_comparison and params.expected_columns is not None:
            result["comparison"] = compare_columns(column_names, params.expected_columns)
        return result

    datadict = DataDictionary()
    return datadict.get_name_labels(
        FileInfo(file_path=file_path),
        expected_columns=params.expected_columns,
        include_file_info=params.include_file_info,
        include_comparison=params.include_comparison,
        columns_only=params.columns_only,
    )



@app.post("/data-dictionary")
async def data_dictionary(fileinfo: FileInfo):
    file_path = resolve_safe_path_http(fileinfo.file_path, label="file_path")
    datadict=DataDictionary()
    return datadict.get_data_dictionary(fileinfo.model_copy(update={"file_path": file_path}))
    


@app.post("/data-dictionary-variable")
async def data_dictionary_variable(params: DictParams):
    file_path = resolve_safe_path_http(params.file_path, label="file_path")
    params = params.model_copy(update={"file_path": file_path})

    file_ext=os.path.splitext(file_path)[1]

    if file_ext.lower() == '.csv':
        datadict=DataDictionaryCsv()
    else:
        datadict=DataDictionary()

    return datadict.get_data_dictionary_variable(params)



@app.post("/generate-csv")
async def write_csv(fileinfo: FileInfo):
    try:
        return write_csv_file(fileinfo)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    


def convert_mixed_column(series):
    def try_convert(x):
        try:
            # Also handles floats that are whole numbers (e.g., 18.0 -> 18)
            if isinstance(x, float) and x.is_integer():
                return int(x)
            return int(str(x)) if str(x).lstrip("-").isdigit() else x
        except (ValueError, TypeError):
            return x

    return series.apply(try_convert)

def write_csv_file(fileinfo: FileInfo):
    file_path = resolve_safe_path(fileinfo.file_path)
    fileinfo = fileinfo.model_copy(update={"file_path": file_path})

    file_ext=os.path.splitext(file_path)[1]
    folder_path=os.path.dirname(file_path)


    try:

        if file_ext.lower() == '.dta':
            csv_filepath = os.path.join(
                folder_path,
                os.path.splitext(os.path.basename(fileinfo.file_path))[0] + '.csv',
            )
            write_dta_to_csv(fileinfo.file_path, csv_filepath, user_missing=True)

        elif file_ext == '.sav':
            # Try multiple encodings for robust SAV file reading
            encodings_to_try = [None, "utf-8", "latin1", "cp1252", "iso-8859-1", "cp850"]
            df, meta = None, None
            last_error = None
            
            for encoding in encodings_to_try:
                try:
                    logger.debug("Trying to read SAV file with encoding: %s", encoding)
                    df, meta = pyreadstat.read_sav(fileinfo.file_path, encoding=encoding, user_missing=True)
                    logger.debug("Successfully read SAV file with encoding: %s", encoding)
                    break
                except (pyreadstat.ReadstatError, UnicodeDecodeError, ValueError) as e:
                    logger.debug("Failed to read SAV with encoding %s: %s", encoding, e)
                    last_error = e
                    continue
            
            # If all encodings failed, try without user_missing=True as fallback
            if df is None:
                logger.debug("All encodings failed with user_missing=True, trying without user_missing...")
                for encoding in encodings_to_try:
                    try:
                        logger.debug("Trying to read SAV file with encoding: %s (user_missing=False)", encoding)
                        df, meta = pyreadstat.read_sav(fileinfo.file_path, encoding=encoding, user_missing=False)
                        logger.debug("Successfully read SAV file with encoding: %s (user_missing=False)", encoding)
                        break
                    except (pyreadstat.ReadstatError, UnicodeDecodeError, ValueError) as e:
                        logger.debug("Failed to read SAV with encoding %s (user_missing=False): %s", encoding, e)
                        last_error = e
                        continue
            
            if df is None:
                raise Exception(f"Failed to read SAV file with any encoding. Last error: {str(last_error)}")

            # CSV has no types; to_csv stringifies values as read from pyreadstat.
            # convert_dtypes / convert_mixed_column are for Stata/SPSS re-export, not CSV.
            # df = df.convert_dtypes()
            # for col in df.columns:
            #     if col in meta.missing_user_values:
            #         df[col] = convert_mixed_column(df[col])
            #         print(f"Converted mixed column: {col}", df[col].dtype)

            csv_filepath = os.path.join(
                folder_path,
                os.path.splitext(os.path.basename(fileinfo.file_path))[0] + '.csv',
            )
            df.to_csv(csv_filepath, index=False)
        else:
            return {"error": "file not supported" + file_ext}

    except Exception as e:
        raise HTTPException(status_code=400, detail="error writing csv file: " + str(e))
    
    output = {
        'status':'success',
        'csv_file':csv_filepath,
        'csv_file_size': DataUtils.sizeof_fmt(os.path.getsize(csv_filepath))      
    }

    return output


def remove_columns_from_csv(params: RemoveColumnsParams) -> Dict[str, Any]:
    """
    Read a CSV, drop the given columns, and write to output_path.
    If output_path already exists, it is overwritten. Caller is responsible for replacing the original file if desired.
    """
    file_path, output_path = resolve_safe_paths(params.file_path, params.output_path, label="file_path")
    params = params.model_copy(update={"file_path": file_path, "output_path": output_path})
    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found: " + file_path)

    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext != ".csv":
        raise ValueError("Source file must be a CSV: " + file_path)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(file_path)
    original_columns = list(df.columns)
    # Drop only columns that exist; ignore missing names
    to_drop = [c for c in params.column_names if c in df.columns]
    df = df.drop(columns=to_drop, errors="ignore")
    df.to_csv(output_path, index=False)

    return {
        "status": "success",
        "output_path": output_path,
        "rows": len(df),
        "columns_remaining": len(df.columns),
        "columns_removed": to_drop,
        "columns_requested_not_found": [c for c in params.column_names if c not in original_columns],
        "output_file_size": DataUtils.sizeof_fmt(os.path.getsize(output_path)),
    }


def detect_column_types(df,meta):
    
    if meta.number_rows > 20000:
        df_sample=df.sample(n=5000, random_state=1)
        df_types=df_sample.convert_dtypes()
    else:        
        df_types=df.convert_dtypes()
    
    return df_types.dtypes.to_dict()


def sanitize_jsonable(obj):
    """Recursively replace NaN/inf values with None for JSON safety."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, np.generic):
        return sanitize_jsonable(obj.item())
    if isinstance(obj, dict):
        return {k: sanitize_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_jsonable(v) for v in obj]
    return obj




async def fifo_worker():
    logger.debug("Starting FIFO worker")

    await recover_pending_jobs(app)

    while True:
        job = await app.fifo_queue.get()
        logger.debug("FIFO worker dequeuing (remaining=%s)", app.fifo_queue.qsize())
        try:
            await job()
        except Exception:
            logger.exception("Unhandled exception in background job")


async def periodic_cleanup_worker():
    """Background task to clean up old jobs every few hours"""
    logger.debug(
        "Starting periodic cleanup worker - will run every %s hours",
        CLEANUP_INTERVAL_HOURS,
    )

    while True:
        await asyncio.sleep(3600 * CLEANUP_INTERVAL_HOURS)
        try:
            logger.debug("Running periodic job cleanup...")
            await cleanup_old_jobs()
        except Exception:
            logger.exception("Cleanup error")



async def cleanup_old_jobs():
    """Remove jobs based on age and status policies"""
    start_time = datetime.datetime.now()
    current_time = start_time
    jobs_to_remove = []
    files_removed = 0
    
    # Cleanup policies by job status
    cleanup_policies = {
        "queued": {"max_age_hours": 2},       # Remove stuck queued jobs after 2 hours
        "waiting": {"max_age_hours": 2},      # Reviewer jobs waiting on semaphore
        "processing": {"max_age_hours": 8},   # Remove stuck processing jobs after 8 hours  
        "done": {"max_age_hours": MAX_JOB_AGE_HOURS},        # Keep completed jobs for configured time
        "error": {"max_age_hours": MAX_JOB_AGE_HOURS * 2},    # Keep error jobs longer for debugging
        "cancelled": {"max_age_hours": 1}     # Remove cancelled jobs after 1 hour
    }
    
    logger.debug("Starting cleanup - current job count: %s", len(app.jobs))
    
    # Find jobs to remove based on age and status
    for jobid, job in app.jobs.items():
        try:
            # Parse created_at timestamp
            if "created_at" not in job:
                # Handle old jobs without timestamps - remove them if they're completed
                if job["status"] in ["done", "error"]:
                    jobs_to_remove.append(jobid)
                continue
                
            created_at = datetime.datetime.fromisoformat(job["created_at"])
            age_hours = (current_time - created_at).total_seconds() / 3600
            
            # Apply cleanup policy based on job status
            job_status = job["status"]
            if job_status in cleanup_policies:
                max_age = cleanup_policies[job_status]["max_age_hours"]
                if age_hours > max_age:
                    jobs_to_remove.append(jobid)
                    logger.debug(
                        "Marking job %s for removal - status: %s, age: %.2fh",
                        jobid,
                        job_status,
                        age_hours,
                    )

        except Exception as e:
            logger.error("Error processing job %s during cleanup: %s", jobid, e)
            # If we can't process the job metadata, remove it if it's old enough
            jobs_to_remove.append(jobid)
    
    # Remove jobs from memory and corresponding files
    for jobid in jobs_to_remove:
        try:
            job = app.jobs.get(jobid, {})
            dispose_reviewer_job_if_needed(app, jobid, job)
            # Remove job file if it exists
            file_path = os.path.join('jobs', f'{jobid}.json')
            if os.path.exists(file_path):
                os.remove(file_path)
                files_removed += 1
            
            # Remove from memory and durable store
            del app.jobs[jobid]
            app.job_store.delete_job(jobid)

        except Exception as e:
            logger.error("Error removing job %s: %s", jobid, e)
    
    # Enforce memory limits (LRU-style cleanup)
    if len(app.jobs) > MAX_MEMORY_JOBS:
        await enforce_memory_limits(jobs_to_remove)
    
    # Clean up orphaned job files
    await cleanup_orphaned_files()
    
    # Update metrics
    cleanup_duration = (datetime.datetime.now() - start_time).total_seconds()
    cleanup_metrics.last_cleanup = current_time.isoformat()
    cleanup_metrics.jobs_cleaned_total += len(jobs_to_remove)
    cleanup_metrics.files_removed_total += files_removed
    cleanup_metrics.cleanup_duration_seconds = cleanup_duration
    
    logger.debug(
        "Cleanup completed - removed %s jobs, %s files in %.2fs",
        len(jobs_to_remove),
        files_removed,
        cleanup_duration,
    )
    logger.debug("Remaining job count: %s", len(app.jobs))


async def enforce_memory_limits(already_removing):
    """Ensure job dictionary doesn't exceed memory limits"""
    if len(app.jobs) <= MAX_MEMORY_JOBS:
        return
        
    # Sort jobs by last_accessed (if available) or created_at, keeping recent and processing jobs
    jobs_with_priority = []
    
    for jobid, job in app.jobs.items():
        if jobid in already_removing:
            continue
            
        # Assign priority - lower number = higher priority (keep longer)
        priority = 5  # default
        
        if job["status"] == "processing":
            priority = 1  # highest priority - never remove processing jobs
        elif job["status"] in ("queued", "waiting"):
            priority = 2  # high priority - keep queued / waiting jobs
        elif job["status"] == "error":
            priority = 4  # lower priority for error jobs
        else:  # done
            priority = 5  # lowest priority for completed jobs
            
        # Use last_accessed if available, otherwise created_at
        timestamp_str = job.get("last_accessed", job.get("created_at"))
        if timestamp_str:
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp_str)
            except:
                timestamp = datetime.datetime.min
        else:
            timestamp = datetime.datetime.min
            
        jobs_with_priority.append((priority, timestamp, jobid))
    
    # Sort by priority (ascending), then by timestamp (ascending = oldest first)
    jobs_with_priority.sort(key=lambda x: (x[0], x[1]))
    
    # Remove oldest, lowest priority jobs until we're under the limit
    jobs_to_remove_for_memory = []
    target_removal_count = len(app.jobs) - MAX_MEMORY_JOBS
    
    for priority, timestamp, jobid in jobs_with_priority:
        if len(jobs_to_remove_for_memory) >= target_removal_count:
            break
        if priority > 2:  # Don't remove processing, queued, or waiting jobs for memory limits
            jobs_to_remove_for_memory.append(jobid)
    
    # Remove the selected jobs
    for jobid in jobs_to_remove_for_memory:
        try:
            job = app.jobs.get(jobid, {})
            dispose_reviewer_job_if_needed(app, jobid, job)
            # Remove job file if it exists
            file_path = os.path.join('jobs', f'{jobid}.json')
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Remove from memory
            del app.jobs[jobid]
            cleanup_metrics.jobs_cleaned_total += 1
            
        except Exception as e:
            logger.error("Error removing job %s for memory limit: %s", jobid, e)

    if jobs_to_remove_for_memory:
        logger.debug("Removed %s jobs to enforce memory limit", len(jobs_to_remove_for_memory))


async def cleanup_orphaned_files():
    """Remove job files that no longer have corresponding entries in app.jobs"""
    jobs_folder = os.path.join(os.getcwd(), 'jobs')
    if not os.path.exists(jobs_folder):
        return
        
    try:
        files = glob.glob(os.path.join(jobs_folder, '*.json'))
        orphaned_files = []
        
        for file_path in files:
            filename = os.path.basename(file_path)
            jobid = filename[:-5]  # Remove .json extension
            
            if jobid not in app.jobs and app.job_store.get_job(jobid) is None:
                orphaned_files.append(file_path)
        
        # Remove orphaned files
        for file_path in orphaned_files:
            try:
                os.remove(file_path)
                cleanup_metrics.files_removed_total += 1
            except Exception as e:
                logger.error("Error removing orphaned file %s: %s", file_path, e)

        if orphaned_files:
            logger.debug("Removed %s orphaned job files", len(orphaned_files))

    except Exception as e:
        logger.error("Error during orphaned file cleanup: %s", e)





@app.on_event("startup")
async def start_background_tasks():
    install_asyncio_exception_handler()
    logger.info(
        "Application starting pid=%s version=%s",
        os.getpid(),
        get_version(),
    )
    asyncio.create_task(fifo_worker())
    asyncio.create_task(periodic_cleanup_worker())


@app.on_event("shutdown")
async def shutdown_tasks():
    logger.info("Application shutting down gracefully")


@app.post("/data-dictionary-queue")
async def data_dictionary_queue(params: DictParams):
    file_path = resolve_safe_path_http(params.file_path, label="file_path")
    params = params.model_copy(update={"file_path": file_path})
    jobid='job-' + str(time.time())
    current_time = datetime.datetime.now().isoformat()
    app.jobs[jobid]={
            "jobid":jobid,
            "jobtype":"data-dictionary",
            "status":"queued",
            "created_at": current_time,
            "completed_at": None,
            "last_accessed": current_time,
            "info":params
        }
    
    data_dict_callback = functools.partial(write_data_dictionary_file, jobid, params)
    await enqueue_fifo_job(app, jobid, data_dict_callback)

    return JSONResponse(status_code=202, content={
        "message": "Item is queued",
        "job_id": jobid
        })



@app.post("/generate-csv-queue")
async def write_csv_queue(fileinfo: FileInfo):

    jobid='job-' + str(time.time())
    current_time = datetime.datetime.now().isoformat()
    app.jobs[jobid]={
            "jobid":jobid,
            "jobtype":"generate-csv",
            "status":"queued",
            "created_at": current_time,
            "completed_at": None,
            "last_accessed": current_time,
            "info":fileinfo
        }
    
    generate_csv_callback=functools.partial(write_csv_file_callback, jobid, fileinfo)
    await enqueue_fifo_job(app, jobid, generate_csv_callback)

    return JSONResponse(status_code=202, content={
        "message": "file is queued",
        "job_id": jobid
        })

    

async def write_csv_file_callback(jobid, fileinfo: FileInfo):
    loop = asyncio.get_running_loop()
    app.jobs[jobid]["status"]="processing"

    try:
        result=await loop.run_in_executor(None, write_csv_file, fileinfo)
    except Exception as e:
        logger.exception("Exception writing csv file for job %s", jobid)
        app.jobs[jobid]["status"]="error"
        app.jobs[jobid]["error"]="failed to write csv file: " + str(e)
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        return {"status":"failed"}


    app.jobs[jobid]["status"]="done"
    app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
    file_path=os.path.join('jobs', str(jobid) + '.json')
    with open(file_path, 'w') as outfile:
        json.dump(result, outfile)
        
    return {"status": "success", "file_path": file_path}


@app.post("/remove-csv-columns-queue")
async def remove_csv_columns_queue(params: RemoveColumnsParams):
    """Queue a job to remove specified columns from a CSV and write the result to a new file. If output_path exists, it is overwritten."""
    file_path, output_path = resolve_safe_paths_http(params.file_path, params.output_path, label="file_path")
    params = params.model_copy(update={"file_path": file_path, "output_path": output_path})
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found: " + file_path)
    if os.path.splitext(file_path)[1].lower() != ".csv":
        raise HTTPException(status_code=400, detail="Source file must be a CSV: " + file_path)

    jobid = "job-" + str(time.time())
    current_time = datetime.datetime.now().isoformat()
    app.jobs[jobid] = {
        "jobid": jobid,
        "jobtype": "remove-csv-columns",
        "status": "queued",
        "created_at": current_time,
        "completed_at": None,
        "last_accessed": current_time,
        "info": params.model_dump(),
    }
    callback = functools.partial(remove_columns_from_csv_callback, jobid, params)
    await enqueue_fifo_job(app, jobid, callback)

    return JSONResponse(status_code=202, content={
        "message": "Remove CSV columns job is queued",
        "job_id": jobid,
    })


async def remove_columns_from_csv_callback(jobid, params: RemoveColumnsParams):
    loop = asyncio.get_running_loop()
    app.jobs[jobid]["status"] = "processing"
    try:
        result = await loop.run_in_executor(None, remove_columns_from_csv, params)
        app.jobs[jobid]["status"] = "done"
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        file_path = os.path.join("jobs", str(jobid) + ".json")
        with open(file_path, "w") as outfile:
            json.dump(result, outfile)
        return {"status": "success", "file_path": file_path}
    except Exception as e:
        logger.error(f"Remove CSV columns failed for job {jobid}: {str(e)}")
        app.jobs[jobid]["status"] = "error"
        app.jobs[jobid]["error"] = str(e)
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        return {"status": "error", "error": str(e)}
    
    

async def write_data_dictionary_file(jobid, params: DictParams):
    file_path = resolve_safe_path(params.file_path)
    params = params.model_copy(update={"file_path": file_path})
    loop = asyncio.get_running_loop()
    file_ext=os.path.splitext(file_path)[1]

    if file_ext.lower() == '.csv':
        datadict=DataDictionaryCsv()
    else:
        datadict=DataDictionary()

    app.jobs[jobid]["status"]="processing"

    try:
        result=await loop.run_in_executor(None, datadict.get_data_dictionary_variable, params)

        app.jobs[jobid]["status"]="done"
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        file_path=os.path.join('jobs', str(jobid) + '.json')
        with open(file_path, 'w') as outfile:
            json.dump(result, outfile)
        
        return {"status": "success", "file_path": file_path}

    except HTTPException as e:
        app.jobs[jobid]["status"]="error"
        detail = e.detail
        err_msg = detail if isinstance(detail, str) else str(detail)
        app.jobs[jobid]["error"] = err_msg
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        app.jobs[jobid]["traceback"] = traceback.format_exc()
        return {"status": "error", "error": err_msg}

    except Exception as e:
        app.jobs[jobid]["status"]="error"
        app.jobs[jobid]["error"] = str(e)
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        app.jobs[jobid]["traceback"] = traceback.format_exc()
        return {"status": "error", "error": str(e)}


@app.post("/export-data-queue")
async def export_data_queue(params: DictParams):
    file_path = resolve_safe_path_http(params.file_path, label="file_path")
    params = params.model_copy(update={"file_path": file_path})
    #print ("export_data_queue", params)
    jobid='job-' + str(time.time())
    current_time = datetime.datetime.now().isoformat()
    app.jobs[jobid]={
            "jobid":jobid,
            "jobtype":"data-export",
            "status":"queued",            
            "created_at": current_time,
            "completed_at": None,
            "last_accessed": current_time,
            "info":params
        }
    
    data_export_callback = functools.partial(export_data_file, jobid, params)
    await enqueue_fifo_job(app, jobid, data_export_callback)

    return JSONResponse(status_code=202, content={
        "message": "Item is queued",
        "job_id": jobid
        })


@app.post("/process-microdata-queue")
async def process_microdata_queue(params: DataProcessingParams):
    """Unified endpoint to process microdata files (CSV generation + data dictionary)"""
    file_path = resolve_safe_path_http(params.file_path, label="file_path")
    params = params.model_copy(update={"file_path": file_path})
    jobid='job-' + str(time.time())
    current_time = datetime.datetime.now().isoformat()
    app.jobs[jobid]={
            "jobid":jobid,
            "jobtype":"process-microdata",
            "status":"queued",
            "created_at": current_time,
            "completed_at": None,
            "last_accessed": current_time,
            "info":params
        }
    
    process_microdata_callback = functools.partial(process_microdata_file, jobid, params)
    await enqueue_fifo_job(app, jobid, process_microdata_callback)

    return JSONResponse(status_code=202, content={
        "message": "Microdata processing is queued",
        "job_id": jobid
        })


async def export_data_file(jobid, params: DictParams):
    file_path = resolve_safe_path(params.file_path)
    params = params.model_copy(update={"file_path": file_path})
    loop = asyncio.get_running_loop()
    file_ext=os.path.splitext(file_path)[1]

    exportDF=ExportDatafile()    
    app.jobs[jobid]["status"]="processing"

    try:
        logger.info(
            "Export job processing: jobid=%s file=%s format=%s",
            jobid,
            params.file_path,
            params.export_format,
        )
        logger.debug(f"Starting export for job {jobid} with params: {params}")
        
        result=await loop.run_in_executor(None, exportDF.export_file, params)

        app.jobs[jobid]["status"]="done"
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        file_path=os.path.join('jobs', str(jobid) + '.json')
        with open(file_path, 'w') as outfile:
            json.dump(result, outfile)
        
        logger.debug(f"Export completed successfully for job {jobid}")
        return {"status": "success", "file_path": file_path}
    
    except Exception as e:
        # Capture detailed error information (no traceback in response/log payloads)
        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "function": "export_data_file",
            "jobid": jobid,
            "params": {
                "file_path": params.file_path,
                "var_names": params.var_names,
                "weights": params.weights,
                "missings": params.missings,
                "dtypes": params.dtypes,
                "value_labels": params.value_labels,
                "export_format": params.export_format,
                "export_options": params.export_options,
            },
        }

        # Log concise error; full traceback is suppressed for API/terminal output
        logger.error(f"Export failed for job {jobid}: {error_info}")
        
        app.jobs[jobid]["status"]="error"
        app.jobs[jobid]["error"]=str(e)
        app.jobs[jobid]["error_details"]={
            "error_type": error_info["error_type"],
            "error_message": error_info["error_message"],
            "function": error_info["function"],
        }
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        return {
            "status": "error",
            "error": str(e),
            "error_details": app.jobs[jobid]["error_details"],
        }


async def process_microdata_file(jobid, params: DataProcessingParams):
    """Process microdata file with both CSV generation and data dictionary creation"""
    file_path = resolve_safe_path(params.file_path)
    params = params.model_copy(update={"file_path": file_path})
    loop = asyncio.get_running_loop()
    app.jobs[jobid]["status"] = "processing"
    
    results = {
        "csv_generation": None,
        "data_dictionary": None,
        "status": "success",
        "processing_steps": []
    }
    
    try:
        logger.debug(f"Starting microdata processing for job {jobid} with params: {params}")
        
        # Step 1: Generate CSV if requested
        if params.generate_csv:
            try:
                logger.debug(f"Job {jobid}: Starting CSV generation")
                app.jobs[jobid]["current_step"] = "generating_csv"
                results["processing_steps"].append("csv_generation_started")
                
                fileinfo = FileInfo(file_path=params.file_path)
                csv_result = await loop.run_in_executor(None, write_csv_file, fileinfo)
                results["csv_generation"] = csv_result
                results["processing_steps"].append("csv_generation_completed")
                
                logger.debug(f"Job {jobid}: CSV generation completed")
                
            except Exception as e:
                error_msg = f"CSV generation failed: {str(e)}"
                logger.error(f"Job {jobid}: {error_msg}")
                results["csv_generation"] = {"status": "error", "error": error_msg}
                results["processing_steps"].append("csv_generation_failed")
                
                # If CSV generation fails and data dictionary requires CSV, fail the whole job
                file_ext = os.path.splitext(params.file_path)[1].lower()
                if params.generate_data_dictionary and file_ext == '.csv':
                    raise Exception(f"Cannot generate data dictionary for CSV file after CSV generation failed: {error_msg}")
        
        # Step 2: Generate data dictionary if requested
        if params.generate_data_dictionary:
            try:
                logger.debug(f"Job {jobid}: Starting data dictionary generation")
                app.jobs[jobid]["current_step"] = "generating_data_dictionary"
                results["processing_steps"].append("data_dictionary_started")
                
                # Convert DataProcessingParams to DictParams for compatibility
                dict_params = DictParams(
                    file_path=params.file_path,
                    var_names=params.var_names,
                    weights=params.weights,
                    missings=params.missings,
                    dtypes=params.dtypes,
                    value_labels=params.value_labels,
                    name_labels=params.name_labels,
                    categorical=params.categorical,
                    export_format=params.export_format
                )
                
                file_ext = os.path.splitext(params.file_path)[1]
                if file_ext.lower() == '.csv':
                    datadict = DataDictionaryCsv()
                else:
                    datadict = DataDictionary()
                
                dict_result = await loop.run_in_executor(None, datadict.get_data_dictionary_variable, dict_params)
                results["data_dictionary"] = dict_result
                results["processing_steps"].append("data_dictionary_completed")
                
                logger.debug(f"Job {jobid}: Data dictionary generation completed")
                
            except HTTPException as e:
                detail = e.detail
                err_detail = detail if isinstance(detail, str) else str(detail)
                error_msg = f"Data dictionary generation failed: {err_detail}"
                logger.error(f"Job {jobid}: {error_msg}")
                results["data_dictionary"] = {"status": "error", "error": error_msg}
                results["processing_steps"].append("data_dictionary_failed")
            except Exception as e:
                error_msg = f"Data dictionary generation failed: {str(e)}"
                logger.error(f"Job {jobid}: {error_msg}")
                results["data_dictionary"] = {"status": "error", "error": error_msg}
                results["processing_steps"].append("data_dictionary_failed")
        
        # Determine overall status
        csv_failed = params.generate_csv and results["csv_generation"] and results["csv_generation"].get("status") == "error"
        dict_failed = params.generate_data_dictionary and results["data_dictionary"] and results["data_dictionary"].get("status") == "error"
        
        if csv_failed and dict_failed:
            results["status"] = "error"
        elif csv_failed or dict_failed:
            results["status"] = "partial_success"
        
        app.jobs[jobid]["status"] = "done"
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        app.jobs[jobid].pop("current_step", None)  # Remove current_step from final result
        
        # Save results to file
        file_path = os.path.join('jobs', str(jobid) + '.json')
        with open(file_path, 'w') as outfile:
            json.dump(results, outfile)
        
        logger.debug(f"Microdata processing completed for job {jobid}")
        return {"status": "success", "file_path": file_path}
    
    except Exception as e:
        # Capture detailed error information
        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "function": "process_microdata_file",
            "jobid": jobid,
            "params": {
                "file_path": params.file_path,
                "generate_csv": params.generate_csv,
                "generate_data_dictionary": params.generate_data_dictionary,
                "var_names": params.var_names,
                "export_format": params.export_format
            }
        }
        
        logger.error(f"Microdata processing failed for job {jobid}: {error_info}")
        
        app.jobs[jobid]["status"] = "error"
        app.jobs[jobid]["error"] = str(e)
        app.jobs[jobid]["error_details"] = error_info
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        app.jobs[jobid].pop("current_step", None)  # Remove current_step from final result
        return {"status": "error", "error": str(e), "error_details": error_info}


@app.get("/jobs")
async def queue_items():
    return {
            "queue_size": app.fifo_queue.qsize(),
            "active_jobs": app.jobs
            }


@app.get("/admin/cleanup-status")
async def cleanup_status():
    """Get cleanup metrics and current system status"""
    return {
        "cleanup_metrics": {
            "last_cleanup": cleanup_metrics.last_cleanup,
            "jobs_cleaned_total": cleanup_metrics.jobs_cleaned_total,
            "files_removed_total": cleanup_metrics.files_removed_total,
            "cleanup_duration_seconds": cleanup_metrics.cleanup_duration_seconds
        },
        "current_status": {
            "job_count": len(app.jobs),
            "queue_size": app.fifo_queue.qsize(),
            "queued_in_store": app.job_store.count_by_status("queued"),
            "max_memory_jobs": MAX_MEMORY_JOBS,
            "max_job_age_hours": MAX_JOB_AGE_HOURS,
            "cleanup_interval_hours": CLEANUP_INTERVAL_HOURS
        },
        "job_status_breakdown": {
            status: len([job for job in app.jobs.values() if job["status"] == status])
            for status in ["queued", "waiting", "processing", "done", "error", "cancelled"]
        }
    }


@app.post("/admin/cleanup-now")
async def manual_cleanup():
    """Manually trigger job cleanup"""
    try:
        await cleanup_old_jobs()
        return {"status": "success", "message": "Manual cleanup completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


def _get_job_record(jobid: str) -> dict | None:
    if jobid in app.jobs:
        return app.jobs[jobid]
    stored = app.job_store.get_job(jobid)
    if stored:
        app.jobs[jobid] = stored
    return stored


@app.get("/jobs/{jobid}")
async def queue_items(jobid: str):
    job = _get_job_record(jobid)
    if job:
        # Update last_accessed timestamp
        job["last_accessed"] = datetime.datetime.now().isoformat()
        app.job_store.update_status(jobid, job["status"], touch_accessed=True)

        if (job["status"]=="done"):
            data={}
            file_path=os.path.join('jobs', str(jobid) + '.json')
            if os.path.exists(file_path):
                with open(file_path) as json_file:
                    data = json.load(json_file)
            else:
                raise HTTPException(status_code=400, detail="Failed to load job data") 

            job_response = sanitize_jsonable(job.copy())
            job_response['data'] = sanitize_jsonable(data)
            return job_response
        elif (job["status"]=="error"):
            logger.debug("Job error response for %s: %s", jobid, job.get("error"))
            # Include detailed error information if available
            if 'error_details' in job:
                error_detail = f"{job['error']}\n\nDetailed Error Information:\n{json.dumps(job['error_details'], indent=2)}"
                raise HTTPException(status_code=400, detail=error_detail)
            else:
                raise HTTPException(status_code=400, detail=job['error'])
        else:
            return sanitize_jsonable(job)

    raise HTTPException(status_code=404, detail="Job not found")


@app.delete("/jobs/{jobid}")
async def cancel_job(jobid: str):
    """
    Cancel a queued or processing job
    
    Args:
        jobid: The unique job identifier
        
    Returns:
        Success message with cancellation details
    """
    job = _get_job_record(jobid)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job["status"]
    current_time = datetime.datetime.now().isoformat()

    # Metadata reviewer: cooperative cancel via threading.Event + asyncio task cancel
    if status in ("queued", "waiting", "processing"):
        dispose_reviewer_job_if_needed(app, jobid, job)
        if job.get("jobtype") == "metadata-reviewer":
            job["status"] = "cancelled"
            job["cancelled_at"] = current_time
            job["cancellation_reason"] = "User requested cancellation"
            logger.info("Reviewer job %s cancelled (was %s)", jobid, status)
            app.job_store.update_status(
                jobid,
                "cancelled",
                cancelled_at=current_time,
                cancellation_reason="User requested cancellation",
                completed_at=current_time,
                touch_accessed=True,
            )
            return {
                "status": "success",
                "message": f"Job {jobid} has been cancelled",
                "job_id": jobid,
                "previous_status": status,
                "cancelled_at": current_time,
                "cancellation_reason": "User requested cancellation",
            }
    
    # Check if job can be cancelled
    if status in ["done", "error", "cancelled"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot cancel job with status '{status}'. Job is already completed or cancelled."
        )
    
    # Cancel the job
    if status == "processing":
        # Mark as cancelled - note: actual task cancellation would require 
        # more complex implementation to stop running tasks
        job["status"] = "cancelled"
        job["cancelled_at"] = current_time
        job["cancellation_reason"] = "User requested cancellation"
        logger.info(f"Job {jobid} marked as cancelled (was processing)")
        
    elif status == "queued":
        # Mark as cancelled; FIFO wrapper skips if still waiting in asyncio queue
        job["status"] = "cancelled"
        job["cancelled_at"] = current_time
        job["cancellation_reason"] = "User requested cancellation"
        logger.info(f"Job {jobid} cancelled (was queued)")

    else:
        # Handle any other status
        job["status"] = "cancelled"
        job["cancelled_at"] = current_time
        job["cancellation_reason"] = "User requested cancellation"
        logger.info(f"Job {jobid} cancelled (was {status})")

    app.job_store.update_status(
        jobid,
        "cancelled",
        cancelled_at=current_time,
        cancellation_reason="User requested cancellation",
        completed_at=current_time,
        touch_accessed=True,
    )

    return {
        "status": "success",
        "message": f"Job {jobid} has been cancelled",
        "job_id": jobid,
        "previous_status": status,
        "cancelled_at": current_time,
        "cancellation_reason": "User requested cancellation"
    }


def remove_jobs_folder():
    folder_path=os.path.join(os.getcwd(), 'jobs')
    if os.path.exists(folder_path):        
        files = glob.glob(folder_path + '/*.json')
        for f in files:
            os.remove(f)


#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)