from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pyreadstat
import time
from src.DataUtils import DataUtils
from src.DataDictionary import DataDictionary
import re
import pandas as pd
import numpy as np
import os
from pydantic import BaseSettings
import json
from src.DictParams import DictParams



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
    missings: list
    
class VarInfo(BaseModel):
    file_path: str
    var_names: list
    weights: list[WeightsColumns] = []
    missings: list[UserMissings] = []



datadict=DataDictionary()

app = FastAPI()


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
