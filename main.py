#import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pyreadstat
import time
from typing import List
from src.DataUtils import DataUtils
from src.DataDictionary import DataDictionary
from src.DataDictionaryCsv import DataDictionaryCsv
from src.ExportDatafile import ExportDatafile
import re
import pandas as pd
import numpy as np
import os
#from pydantic import BaseSettings
from pydantic_settings import BaseSettings
import json
from src.DictParams import DictParams
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

# Configure logging
def setup_logging():
    """Configure logging based on environment variables"""
    # Get logging configuration from environment variables
    log_level = os.getenv("LOG_LEVEL", "ERROR").upper()
    log_format = os.getenv("LOG_FORMAT", "simple")
    log_to_file = os.getenv("LOG_TO_FILE", "false").lower() == "true"
    
    # Generate default log file path with date-based naming
    if log_to_file:
        # Create logs directory if it doesn't exist
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        # Generate date-based filename
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        default_log_file = os.path.join(logs_dir, f"error-{current_date}.log")
    else:
        default_log_file = "app.log"  # Fallback for when file logging is disabled
    
    log_file_path = os.getenv("LOG_FILE_PATH", default_log_file)
    
    # Convert string log level to logging constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    log_level_constant = level_map.get(log_level, logging.ERROR)
    
    # Define log formats
    formats = {
        "simple": "%(levelname)s - %(message)s",
        "detailed": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        "timestamp": "%(asctime)s - %(levelname)s - %(message)s",
        "minimal": "%(levelname)s: %(message)s"
    }
    
    log_format_string = formats.get(log_format, formats["simple"])
    
    # Configure logging
    if log_to_file:
        # Ensure the directory for the log file exists
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        logging.basicConfig(
            level=log_level_constant,
            format=log_format_string,
            handlers=[
                logging.FileHandler(log_file_path),
                logging.StreamHandler()  # Also log to console
            ]
        )
        print(f"Logging configured: Level={log_level}, Format={log_format}, File={log_file_path}")
    else:
        logging.basicConfig(
            level=log_level_constant,
            format=log_format_string
        )
        print(f"Logging configured: Level={log_level}, Format={log_format}")
    
    return logging.getLogger(__name__)

# Setup logging with configuration
logger = setup_logging()


# Load environment variables from the .env file
load_dotenv(override=True)

# Cleanup configuration
# run cleanup task to remove old jobs
CLEANUP_INTERVAL_HOURS = int(os.getenv("CLEANUP_INTERVAL_HOURS", "1"))
# remove jobs older than this
MAX_JOB_AGE_HOURS = int(os.getenv("MAX_JOB_AGE_HOURS", "24"))

# limit the number of jobs in memory
MAX_MEMORY_JOBS = int(os.getenv("MAX_MEMORY_JOBS", "500"))

storage_path = os.getenv("STORAGE_PATH")
if storage_path is not None:
    if not os.path.exists(storage_path):
        raise ValueError("STORAGE_PATH does not exist: " + storage_path)
    else:
        print("STORAGE_PATH:", storage_path)
else:
    print("STORAGE_PATH not set - path validation disabled")



#class Settings(BaseSettings):
#    storage_path: str = "data"    
    

#settings = Settings()


class FileInfo(BaseModel):
    file_path: str

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



datadict=DataDictionary()

app = FastAPI()
app.fifo_queue = asyncio.Queue()

app.jobs = {}

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

    import traceback
    print(traceback.format_exc())   
    print(f"error: {repr(exc)}")
    return await http_exception_handler(request, exc)



@app.get("/")
async def root(request: Request):
    return {"message": "PyDataTool API - See documentation at " + str(request.url) + "docs"}

@app.get("/status")
async def status():
    return {"status": "ok"}


@app.post("/metadata")
async def metadata(fileinfo: FileInfo):

    datadict=DataDictionary()
    return datadict.get_metadata(fileinfo)

@app.post("/name-labels")
async def name_labels(fileinfo: FileInfo):

    datadict=DataDictionary()
    return datadict.get_name_labels(fileinfo)



@app.post("/data-dictionary")
async def data_dictionary(fileinfo: FileInfo):

    datadict=DataDictionary()
    return datadict.get_data_dictionary(fileinfo)
    


@app.post("/data-dictionary-variable")
async def data_dictionary_variable(params: DictParams):

    file_ext=os.path.splitext(params.file_path)[1]

    if file_ext.lower() == '.csv':
        datadict=DataDictionaryCsv()
    else:
        datadict=DataDictionary()

    return datadict.get_data_dictionary_variable(params)



@app.post("/generate-csv")
async def write_csv(fileinfo: FileInfo):
    return write_csv_file(fileinfo)
    


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
    
    # Check if the file path is safe
    if not is_safe_path(fileinfo.file_path):
        raise HTTPException(status_code=400, detail="Invalid file path: " + fileinfo.file_path)


    file_ext=os.path.splitext(fileinfo.file_path)[1]
    folder_path=os.path.dirname(fileinfo.file_path)


    try:

        if file_ext.lower() == '.dta':
            # Try multiple encodings for robust file reading
            encodings_to_try = [None, "utf-8", "latin1", "cp1252", "iso-8859-1", "cp850"]
            df, meta = None, None
            last_error = None
            
            for encoding in encodings_to_try:
                try:
                    print(f"Trying to read DTA file with encoding: {encoding}")
                    df, meta = pyreadstat.read_dta(fileinfo.file_path, encoding=encoding, user_missing=True)
                    print(f"Successfully read DTA file with encoding: {encoding}")
                    break
                except (pyreadstat.ReadstatError, UnicodeDecodeError, ValueError) as e:
                    print(f"Failed to read with encoding {encoding}: {str(e)}")
                    last_error = e
                    continue
            
            # If all encodings failed, try without user_missing=True as fallback
            if df is None:
                print("All encodings failed with user_missing=True, trying without user_missing...")
                for encoding in encodings_to_try:
                    try:
                        print(f"Trying to read DTA file with encoding: {encoding} (user_missing=False)")
                        df, meta = pyreadstat.read_dta(fileinfo.file_path, encoding=encoding, user_missing=False)
                        print(f"Successfully read DTA file with encoding: {encoding} (user_missing=False)")
                        break
                    except (pyreadstat.ReadstatError, UnicodeDecodeError, ValueError) as e:
                        print(f"Failed to read with encoding {encoding} (user_missing=False): {str(e)}")
                        last_error = e
                        continue
            
            if df is None:
                raise Exception(f"Failed to read DTA file with any encoding. Last error: {str(last_error)}")                

        elif file_ext == '.sav':
            # Try multiple encodings for robust SAV file reading
            encodings_to_try = [None, "utf-8", "latin1", "cp1252", "iso-8859-1", "cp850"]
            df, meta = None, None
            last_error = None
            
            for encoding in encodings_to_try:
                try:
                    print(f"Trying to read SAV file with encoding: {encoding}")
                    df, meta = pyreadstat.read_sav(fileinfo.file_path, encoding=encoding, user_missing=True)
                    print(f"Successfully read SAV file with encoding: {encoding}")
                    break
                except (pyreadstat.ReadstatError, UnicodeDecodeError, ValueError) as e:
                    print(f"Failed to read SAV with encoding {encoding}: {str(e)}")
                    last_error = e
                    continue
            
            # If all encodings failed, try without user_missing=True as fallback
            if df is None:
                print("All encodings failed with user_missing=True, trying without user_missing...")
                for encoding in encodings_to_try:
                    try:
                        print(f"Trying to read SAV file with encoding: {encoding} (user_missing=False)")
                        df, meta = pyreadstat.read_sav(fileinfo.file_path, encoding=encoding, user_missing=False)
                        print(f"Successfully read SAV file with encoding: {encoding} (user_missing=False)")
                        break
                    except (pyreadstat.ReadstatError, UnicodeDecodeError, ValueError) as e:
                        print(f"Failed to read SAV with encoding {encoding} (user_missing=False): {str(e)}")
                        last_error = e
                        continue
            
            if df is None:
                raise Exception(f"Failed to read SAV file with any encoding. Last error: {str(last_error)}")
        else:
            return {"error": "file not supported" + file_ext}
    

        df=df.convert_dtypes()

        # Convert mixed columns to numeric if they contain user-defined missings
        for col in df.columns:
            #check  meta for user-defined missings
            if col in meta.missing_user_values:
                #convert mixed columns to numeric
                df[col] = convert_mixed_column(df[col])
                print(f"Converted mixed column: {col}", df[col].dtype)
                continue


        csv_filepath = os.path.join(folder_path,os.path.splitext(os.path.basename(fileinfo.file_path))[0] + '.csv')    
        df.to_csv(csv_filepath, index=False)

    except Exception as e:
        raise HTTPException(status_code=400, detail="error writing csv file: " + str(e))
    
    output = {
        'status':'success',
        'csv_file':csv_filepath,
        'csv_file_size': DataUtils.sizeof_fmt(os.path.getsize(csv_filepath))      
    }

    return output





def detect_column_types(df,meta):
    
    if meta.number_rows > 20000:
        df_sample=df.sample(n=5000, random_state=1)
        df_types=df_sample.convert_dtypes()
    else:        
        df_types=df.convert_dtypes()
    
    return df_types.dtypes.to_dict()




async def fifo_worker():
    print("Starting FIFO worker")

    # remove old jobs
    remove_jobs_folder()

    while True:
        job = await app.fifo_queue.get()
        print(f"Got a job: (size of remaining queue: {app.fifo_queue.qsize()})")
        await job()


async def periodic_cleanup_worker():
    """Background task to clean up old jobs every few hours"""
    print(f"Starting periodic cleanup worker - will run every {CLEANUP_INTERVAL_HOURS} hours")
    
    while True:
        await asyncio.sleep(3600 * CLEANUP_INTERVAL_HOURS)
        try:
            print("Running periodic job cleanup...")
            await cleanup_old_jobs()
        except Exception as e:
            print(f"Cleanup error: {e}")
            import traceback
            print(traceback.format_exc())


async def cleanup_old_jobs():
    """Remove jobs based on age and status policies"""
    start_time = datetime.datetime.now()
    current_time = start_time
    jobs_to_remove = []
    files_removed = 0
    
    # Cleanup policies by job status
    cleanup_policies = {
        "queued": {"max_age_hours": 2},       # Remove stuck queued jobs after 2 hours
        "processing": {"max_age_hours": 8},   # Remove stuck processing jobs after 8 hours  
        "done": {"max_age_hours": MAX_JOB_AGE_HOURS},        # Keep completed jobs for configured time
        "error": {"max_age_hours": MAX_JOB_AGE_HOURS * 2}    # Keep error jobs longer for debugging
    }
    
    print(f"Starting cleanup - current job count: {len(app.jobs)}")
    
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
                    print(f"Marking job {jobid} for removal - status: {job_status}, age: {age_hours:.2f}h")
            
        except Exception as e:
            print(f"Error processing job {jobid} during cleanup: {e}")
            # If we can't process the job metadata, remove it if it's old enough
            jobs_to_remove.append(jobid)
    
    # Remove jobs from memory and corresponding files
    for jobid in jobs_to_remove:
        try:
            # Remove job file if it exists
            file_path = os.path.join('jobs', f'{jobid}.json')
            if os.path.exists(file_path):
                os.remove(file_path)
                files_removed += 1
            
            # Remove from memory
            del app.jobs[jobid]
            
        except Exception as e:
            print(f"Error removing job {jobid}: {e}")
    
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
    
    print(f"Cleanup completed - removed {len(jobs_to_remove)} jobs, {files_removed} files in {cleanup_duration:.2f}s")
    print(f"Remaining job count: {len(app.jobs)}")


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
        elif job["status"] == "queued":
            priority = 2  # high priority - keep queued jobs
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
        if priority > 2:  # Don't remove processing or queued jobs for memory limits
            jobs_to_remove_for_memory.append(jobid)
    
    # Remove the selected jobs
    for jobid in jobs_to_remove_for_memory:
        try:
            # Remove job file if it exists
            file_path = os.path.join('jobs', f'{jobid}.json')
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Remove from memory
            del app.jobs[jobid]
            cleanup_metrics.jobs_cleaned_total += 1
            
        except Exception as e:
            print(f"Error removing job {jobid} for memory limit: {e}")
    
    if jobs_to_remove_for_memory:
        print(f"Removed {len(jobs_to_remove_for_memory)} jobs to enforce memory limit")


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
            
            if jobid not in app.jobs:
                orphaned_files.append(file_path)
        
        # Remove orphaned files
        for file_path in orphaned_files:
            try:
                os.remove(file_path)
                cleanup_metrics.files_removed_total += 1
            except Exception as e:
                print(f"Error removing orphaned file {file_path}: {e}")
        
        if orphaned_files:
            print(f"Removed {len(orphaned_files)} orphaned job files")
            
    except Exception as e:
        print(f"Error during orphaned file cleanup: {e}")





@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(fifo_worker())
    asyncio.create_task(periodic_cleanup_worker())        


@app.post("/data-dictionary-queue")
async def data_dictionary_queue(params: DictParams):    
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
    await app.fifo_queue.put( data_dict_callback )

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
    await app.fifo_queue.put( generate_csv_callback )

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
        print ("exception writing csv file", e)        
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
    
    

async def write_data_dictionary_file(jobid, params: DictParams):
    loop = asyncio.get_running_loop()
    file_ext=os.path.splitext(params.file_path)[1]

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
    
    except Exception as e:
        import traceback
        app.jobs[jobid]["status"]="error"
        app.jobs[jobid]["error"]=str(e)
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        app.jobs[jobid]["traceback"]=traceback.format_exc()
        return {"status": "error", "error": str(e)}


@app.post("/export-data-queue")
async def export_data_queue(params: DictParams):
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
    await app.fifo_queue.put( data_export_callback )

    return JSONResponse(status_code=202, content={
        "message": "Item is queued",
        "job_id": jobid
        })


async def export_data_file(jobid, params: DictParams):
    loop = asyncio.get_running_loop()
    file_ext=os.path.splitext(params.file_path)[1]

    exportDF=ExportDatafile()    
    app.jobs[jobid]["status"]="processing"

    try:
        # Debug logging (only shown when LOG_LEVEL=DEBUG)
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
        # Capture detailed error information
        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "function": "export_data_file",
            "jobid": jobid,
            "params": {
                "file_path": params.file_path,
                "var_names": params.var_names,
                "weights": params.weights,
                "missings": params.missings,
                "dtypes": params.dtypes,
                "value_labels": params.value_labels,
                "export_format": params.export_format
            }
        }
        
        logger.error(f"Export failed for job {jobid}: {error_info}")
        
        app.jobs[jobid]["status"]="error"
        app.jobs[jobid]["error"]=str(e)
        app.jobs[jobid]["error_details"]=error_info
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
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
            "max_memory_jobs": MAX_MEMORY_JOBS,
            "max_job_age_hours": MAX_JOB_AGE_HOURS,
            "cleanup_interval_hours": CLEANUP_INTERVAL_HOURS
        },
        "job_status_breakdown": {
            status: len([job for job in app.jobs.values() if job["status"] == status])
            for status in ["queued", "processing", "done", "error"]
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


@app.get("/jobs/{jobid}")
async def queue_items(jobid: str):

    if jobid in app.jobs:
        job = app.jobs[jobid]
        
        # Update last_accessed timestamp
        job["last_accessed"] = datetime.datetime.now().isoformat()
        
        if (job["status"]=="done"):
            data={}
            file_path=os.path.join('jobs', str(jobid) + '.json')
            if os.path.exists(file_path):
                with open(file_path) as json_file:
                    data = json.load(json_file)
            else:
                raise HTTPException(status_code=400, detail="Failed to load job data") 

            job_response=job.copy()
            job_response['data']=data            
            return job_response
        elif (job["status"]=="error"):
            print ("job error", job)
            # Include detailed error information if available
            if 'error_details' in job:
                error_detail = f"{job['error']}\n\nDetailed Error Information:\n{json.dumps(job['error_details'], indent=2)}"
                raise HTTPException(status_code=400, detail=error_detail)
            else:
                raise HTTPException(status_code=400, detail=job['error'])
        else:
            return job

    raise HTTPException(status_code=404, detail="Job not found") 


def remove_jobs_folder():
    folder_path=os.path.join(os.getcwd(), 'jobs')
    if os.path.exists(folder_path):        
        files = glob.glob(folder_path + '/*.json')
        for f in files:
            os.remove(f)



def is_safe_path(file_path: str) -> bool:
    """
    Validate that the file path is within the storage directory.
    If STORAGE_PATH is not set, path validation is disabled.

    Args:
        file_path (str): The target file path to validate.

    Returns:
        bool: True if the path is safe, False otherwise.
    """
    # Get the storage path from the environment variable
    storage_path = os.getenv("STORAGE_PATH")

    # If STORAGE_PATH is not set, skip path validation
    if not storage_path:
        return True

    # Resolve and normalize paths
    storage_path = os.path.abspath(os.path.normpath(storage_path))
    target_path = os.path.abspath(os.path.normpath(file_path))

    # Check if the target path is within the storage path
    return target_path.startswith(storage_path)


#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)