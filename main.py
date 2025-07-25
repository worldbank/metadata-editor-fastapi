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

from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from starlette.exceptions import HTTPException as StarletteHTTPException


# Load environment variables from the .env file
load_dotenv(override=True)


if os.getenv("STORAGE_PATH") is None:
    raise ValueError("STORAGE_PATH environment variable is not set")
elif not os.path.exists(os.getenv("STORAGE_PATH")):
    raise ValueError("STORAGE_PATH does not exist: " + os.getenv("STORAGE_PATH"))
else:
    print("STORAGE_PATH:", os.getenv("STORAGE_PATH"))



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





@app.on_event("startup")
async def start_queue():
    asyncio.create_task(fifo_worker())        


@app.post("/data-dictionary-queue")
async def data_dictionary_queue(params: DictParams):    
    jobid='job-' + str(time.time())
    app.jobs[jobid]={
            "jobid":jobid,
            "jobtype":"data-dictionary",
            "status":"queued",
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
    app.jobs[jobid]={
            "jobid":jobid,
            "jobtype":"generate-csv",
            "status":"queued",
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
        return {"status":"failed"}


    app.jobs[jobid]["status"]="done"
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
        file_path=os.path.join('jobs', str(jobid) + '.json')
        with open(file_path, 'w') as outfile:
            json.dump(result, outfile)
        
        return {"status": "success", "file_path": file_path}
    
    except Exception as e:
        import traceback
        app.jobs[jobid]["status"]="error"
        app.jobs[jobid]["error"]=str(e)
        app.jobs[jobid]["traceback"]=traceback.format_exc()
        return {"status": "error", "error": str(e)}


@app.post("/export-data-queue")
async def export_data_queue(params: DictParams):
    print ("export_data_queue", params)
    jobid='job-' + str(time.time())
    app.jobs[jobid]={
            "jobid":jobid,
            "jobtype":"data-export",
            "status":"queued",
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
        result=await loop.run_in_executor(None, exportDF.export_file, params)

        app.jobs[jobid]["status"]="done"
        file_path=os.path.join('jobs', str(jobid) + '.json')
        with open(file_path, 'w') as outfile:
            json.dump(result, outfile)
        
        return {"status": "success", "file_path": file_path}
    
    except Exception as e:
        app.jobs[jobid]["status"]="error"
        app.jobs[jobid]["error"]=str(e)
        return {"status": "error", "error": str(e)}


@app.get("/jobs")
async def queue_items():
    return {
            "queue_size": app.fifo_queue.qsize(),
            "active_jobs": app.jobs
            }


@app.get("/jobs/{jobid}")
async def queue_items(jobid: str):

    if jobid in app.jobs:
        job=app.jobs[jobid]
        
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

    Args:
        file_path (str): The target file path to validate.

    Returns:
        bool: True if the path is safe, False otherwise.
    """
    # Get the storage path from the environment variable
    storage_path = os.getenv("STORAGE_PATH")

    if not storage_path:
        raise ValueError("STORAGE_PATH environment variable is not set")

    # Resolve and normalize paths
    storage_path = os.path.abspath(os.path.normpath(storage_path))
    target_path = os.path.abspath(os.path.normpath(file_path))

    # Check if the target path is within the storage path
    return target_path.startswith(storage_path)


#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)