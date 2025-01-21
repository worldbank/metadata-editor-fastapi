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
from pydantic import BaseSettings
import json
from src.DictParams import DictParams
import asyncio
import functools
import hashlib
import datetime
from fastapi.concurrency import run_in_threadpool
import shutil
import glob

from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from starlette.exceptions import HTTPException as StarletteHTTPException




class Settings(BaseSettings):
    storage_path: str = "data"    
    

settings = Settings()


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
    


def write_csv_file(fileinfo: FileInfo):

    file_ext=os.path.splitext(fileinfo.file_path)[1]    
    folder_path=os.path.dirname(fileinfo.file_path)
    file_exists=os.path.exists(fileinfo.file_path)

    #if not file_exists:
    #    raise HTTPException(status_code=400, detail="file not found: " + fileinfo.file_path)
    
    try:

        if file_ext.lower() == '.dta':
            try:
                df,meta = pyreadstat.read_dta(fileinfo.file_path)
            except pyreadstat.ReadstatError as e:
                df,meta = pyreadstat.read_dta(fileinfo.file_path, encoding="latin1")
            except UnicodeDecodeError as e:
                df,meta = pyreadstat.read_dta(fileinfo.file_path, encoding="latin1")                

        elif file_ext == '.sav':
            df, meta = pyreadstat.read_sav(fileinfo.file_path)
        else:
            return {"error": "file not supported" + file_ext}
    

        df=df.convert_dtypes()
        csv_filepath = os.path.join(folder_path,os.path.splitext(os.path.basename(fileinfo.file_path))[0] + '.csv')    
        df.to_csv(csv_filepath, index=False)

    except Exception as e:
        #print("error-writing-csv================= " + str(e))
        raise HTTPException(status_code=400, detail="error writing csv file: " + str(e))
    
    output = {
        #'path':os.path.abspath(os.getcwd()),
        #'abspath':os.path.dirname(os.path.abspath(__file__)),
        #'filename':os.path.basename(fileinfo.file_path),
        #'file_ext':os.path.splitext(fileinfo.file_path)[1],
        #'file_path':os.path.dirname(fileinfo.file_path),
        #'file_exists':os.path.exists(fileinfo.file_path),
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




#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)