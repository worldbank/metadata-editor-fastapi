from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pyreadstat
import time
from typing import List
from src.DataUtils import DataUtils
from src.DataDictionary import DataDictionary
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



@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/status")
async def status():
    return {"status": "ok"}


@app.post("/metadata")
async def metadata(fileinfo: FileInfo):

    datadict=DataDictionary()
    return datadict.get_metadata(fileinfo)


@app.post("/data-dictionary")
async def data_dictionary(fileinfo: FileInfo):

    datadict=DataDictionary()
    return datadict.get_data_dictionary(fileinfo)
    


@app.post("/data-dictionary-variable")
async def data_dictionary_variable(params: DictParams):

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
            df,meta = pyreadstat.read_dta(fileinfo.file_path)
        elif file_ext == '.sav':
            df, meta = pyreadstat.read_sav(fileinfo.file_path)
        else:
            return {"error": "file not supported" + file_ext}
        
    
        df=df.convert_dtypes()

        #csv_filepath = os.path.join(settings.storage_path, os.path.splitext(os.path.basename(fileinfo.file_path))[0] + '.csv')
        csv_filepath = os.path.join(folder_path,os.path.splitext(os.path.basename(fileinfo.file_path))[0] + '.csv')    
        df.to_csv(csv_filepath, index=False)
    except Exception as e:
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

    result=await loop.run_in_executor(None, write_csv_file, fileinfo)

    app.jobs[jobid]["status"]="done"
    file_path=os.path.join('jobs', str(jobid) + '.json')
    with open(file_path, 'w') as outfile:
        json.dump(result, outfile)
        
    return {"status": "success", "file_path": file_path}
    
    

async def write_data_dictionary_file(jobid, params: DictParams):
    loop = asyncio.get_running_loop()
    datadict=DataDictionary()
    app.jobs[jobid]["status"]="processing"

    result=await loop.run_in_executor(None, datadict.get_data_dictionary_variable, params)

    app.jobs[jobid]["status"]="done"
    file_path=os.path.join('jobs', str(jobid) + '.json')
    with open(file_path, 'w') as outfile:
        json.dump(result, outfile)
        
    return {"status": "success", "file_path": file_path}



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

            job['data']=data
            return job 
        else:
            return job

    raise HTTPException(status_code=404, detail="Job not found") 


def remove_jobs_folder():
    folder_path=os.path.join(os.getcwd(), 'jobs')
    if os.path.exists(folder_path):        
        files = glob.glob(folder_path + '/*.json')
        for f in files:
            os.remove(f)