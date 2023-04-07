from fastapi import FastAPI, HTTPException
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


class Settings(BaseSettings):
    storage_path: str = "data"    
    


settings = Settings()


class FileInfo(BaseModel):
    file_path: str
    
class VarInfo(BaseModel):
    file_path: str
    var_names: list 

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

    file_ext=os.path.splitext(fileinfo.file_path)[1]
    folder_path=os.path.dirname(fileinfo.file_path)
    file_exists=os.path.exists(fileinfo.file_path)

    if not file_exists:
        return {"error": "file not found: " + fileinfo.file_path}

    if file_ext.lower() == '.dta':
        df,meta = pyreadstat.read_dta(fileinfo.file_path, metadataonly=True)
    elif file_ext.lower() == '.sav':
        df, meta = pyreadstat.read_sav(fileinfo.file_path)       
    elif file_ext == '.csv':
        df, meta = pyreadstat.read_csv(fileinfo.file_path, metadataonly=True)
    else:
        return {"error": "file not supported" + file_ext}

    variables=[]

    for name in meta.column_names:
        variables.append(
            {
                'name':name,
                'labl':meta.column_names_to_labels[name],
                'var_intrvl': datadict.variable_measure(meta,name),
                'var_format': datadict.variable_format(meta,name),
                'var_catgry': datadict.variable_categories(meta,name)
            }
        )

    basic_sumstat = {
        #'path':os.path.abspath(os.getcwd()),
        #'abspath':os.path.dirname(os.path.abspath(__file__)),
        #'filename':os.path.basename(fileinfo.file_path),
        #'file_ext':os.path.splitext(fileinfo.file_path)[1],
        #'file_path':os.path.dirname(fileinfo.file_path),
        #'file_exists':os.path.exists(fileinfo.file_path),
        'rows':meta.number_rows,
        'columns':meta.number_columns,
        'variables':variables,        
    }

    return basic_sumstat


@app.post("/data-dictionary")
async def data_dictionary(fileinfo: FileInfo):

    file_ext=os.path.splitext(fileinfo.file_path)[1]
    folder_path=os.path.dirname(fileinfo.file_path)
    file_exists=os.path.exists(fileinfo.file_path)

    if not file_exists:
        return {"error": "file not found: " + fileinfo.file_path}

    if file_ext.lower() == '.dta':
        df, meta = pyreadstat.read_dta(fileinfo.file_path)
    elif file_ext.lower() == '.sav':
        df, meta = pyreadstat.read_sav(fileinfo.file_path)    
    elif file_ext == '.csv':
        df, meta = pyreadstat.read_csv(fileinfo.file_path)
    else:
        return {"error": "file not supported" + file_ext}

    df=df.convert_dtypes()

    variables = []
    for name in meta.column_names:
        #print("type is ",json.dumps(variable_summary(df,meta,name)))
        
        variables.append(datadict.variable_summary(df,meta,name))
        #variables.append()
        
    return {
        'rows':meta.number_rows,
        'columns':meta.number_columns,
        'variables':variables
        }




@app.post("/data-dictionary-variable")
async def data_dictionary(info: VarInfo):

    file_ext=os.path.splitext(info.file_path)[1]
    folder_path=os.path.dirname(info.file_path)
    file_exists=os.path.exists(info.file_path)

    if not file_exists:
        return {"error": "file not found: " + info.file_path}

    if file_ext.lower() == '.dta':
        df, meta = pyreadstat.read_dta(info.file_path, usecols=info.var_names)
    elif file_ext.lower() == '.sav':
        df, meta = pyreadstat.read_sav(info.file_path,usecols=[info.var_names])
    #elif file_ext == '.csv':
    #    df, meta = pyreadstat.read_csv(info.file_path)
    else:
        return {"error": "file not supported" + file_ext}

    df=df.convert_dtypes()

    variables = []
    for name in meta.column_names:
        variables.append(datadict.variable_summary(df,meta,name))
        
    return {
        'rows':meta.number_rows,
        'columns':meta.number_columns,
        'variables':variables
        }







@app.get("/data-dictionary-test")
async def data_dictionary():
    df, meta = pyreadstat.read_dta('data/data.dta',user_missing=True, apply_value_formats=True)
    #return meta.column_names_to_labels
    #df, meta = pyreadstat.read_dta('data/argentina_cos_fy19_datafile_final_.dta',user_missing=True)
    #return df.head().to_json()

    variables = []
    for name in meta.column_names:
        variables.append(datadict.variable_summary(df,meta,name))

    return variables





@app.get("/read_csv")
async def read_csv():
    try:
        df = pd.read_csv('data/SCHOOL5.csv')
        print ("==================finished reading file ===================")
        print (df.head())
        print("-------------------------------------")    
        #print (df.dtypes[df.dtypes == 'category'].index)
        
        print (df.describe(percentiles=None))
        
        print("----------- with nulls--------------------------")            
        print (df.describe(percentiles=None, include='all',))
        print("----------- aggregates--------------------------")    
        print (df.agg(['count', 'size', 'nunique']))

        print("-----------value counts--------------------------")    
        print(df.value_counts())
        print("-----------value counts by column--------------------------")    

        # get column counts/frequencies
        print(df['shcode'].value_counts())
        

        print("-------------------------------------")    
        
        

        #return df.head().to_dict()
    except Exception as e:
        print(e)




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




@app.get("/read")
async def read():

    start_time = time.time()

    #filepath="/Volumes/webdev/editor/datafiles/editor/f7177163c833dff4b38fc8d2872f1ec6/data/ZZBR62FL.DTA"
    filepath='data/argentina_cos_fy19_datafile_final_.dta'
    #filepath="/Volumes/webdev/editor/datafiles/editor/f7177163c833dff4b38fc8d2872f1ec6/data/viviendas.dta"
    df, meta = pyreadstat.read_dta(filepath,user_missing=True, apply_value_formats=False)
    #df, meta = pyreadstat.read_csv('data/SCHOOL4.csv')
    #df,meta = pyreadstat.read_dta('/Users/m2/Downloads/synthethic_data-all-imputed-1665872515-N5000000_clean.dta',
    #                              metadataonly=False, row_offset=1, row_limit=10000 )
    
    #df,meta = pyreadstat.read_dta('/Users/m2/Downloads/synthethic_data-all-imputed-1665872515-N5000000_clean.dta',
    #                            metadataonly=False,usecols=['sex','relation'],row_offset=1, row_limit=10000
    #                              )

    #df,meta = pyreadstat.read_dta('/Users/m2/Downloads/synthethic_data-indiv-imputed-1665872515-N5000000_clean.dta',
    #                            user_missing=True, apply_value_formats=True
    #                              )

    #column_types=detect_column_types(df,meta)
    #print("------ column types ------------------")
    #print(column_types)

    print("------ end column types ------------------")
    print(meta)

    # done! let's see what we got
    
    #print(meta.column_names)
    #print(meta.column_labels)
    print (meta.variable_value_labels  )
    print(meta.column_names_to_labels)
    print("number of rows",meta.number_rows)
    print(meta.number_columns)
    #print(meta.file_label)
    #print(meta.file_encoding)
    df = df.convert_dtypes()
    return 
    
    #df_sub=df[:100]
    #df_c=df.convert_dtypes()
    #print (df.infer_objects().dtypes)
    print (df.dtypes )
    print(df['a1'].value_counts())
    dfn = df.convert_dtypes()
    df=[]
    df=dfn
    
    print("=====================================")

    print (dfn.dtypes )
    
    print(df['a1'].value_counts())

    #get mean, meadian, mode, std, min, max, count, unique, top, freq for all columns
    #print (dfn.describe(percentiles=None))


    return "X"

    #print(df["sex"].value_counts())
    #print(df["relation"].value_counts())
    #return df["relation"].value_counts()

    print("----------- with nulls--------------------------")            
    #print (df.describe(percentiles=None, include='all',))
    print("----------- aggregates--------------------------")    
    print (df['a1'].agg(['count', 'size', 'nunique']))

    print("-----------value counts--------------------------")    
    #print(df.value_counts())
    print("-----------value counts by column--------------------------")    
    print(df['a1'].value_counts())
    #a1=pd.Series(df['a1'], dtype=pd.Int64Dtype())
    
    

    #a1_value_counts=df_c['a1'].value_counts()

    #print (a1_value_counts)

    x=pd.Series(df['a1'], dtype=pd.Int64Dtype())
    print(x.value_counts())

    print (df['a1'].isnull().sum())

    print (df['a1'].describe(percentiles=None, include='all',))

    print (df['a1'].agg(['count', 'size', 'nunique']))
    #print (df['a6_2'].isnull())

    #check if floats are integers
    #print (df['id'].apply(float.is_integer).all())

    #x=pd.Series(df['a6_2'], dtype=pd.Int64Dtype())

    df['a6_2x'] = df['a6_2'].fillna(-1)

    df['a6_2x'] = df['a6_2x'].astype(int)

    df['a6_2xx']=pd.Series(df['a6_2'], dtype=pd.Int64Dtype())

    print (df['a6_2xx'].value_counts())

    print (df['a6_2x'].value_counts())


    print (df.dtypes )
    newdf = df.convert_dtypes()


    print (newdf.dtypes )
    
    
    #print(pd.Series(a1_value_counts, dtype=pd.Int64Dtype()))

    # get column counts/frequencies
    

    #print (df.dtypes)


    basic_sumstat = {
        'rows':meta.number_rows,
        'columns':meta.number_columns,
        #'file_label':meta.file_label,
        #'file_encoding':meta.file_encoding,
        #'column_names_to_labels':meta.column_names_to_labels,
        #'dtypes':df.dtypes.to_dict(),
        'value_counts':df['a1'].value_counts().to_dict(),
    }
    #basic_sumstat = [
    #    {"type": 'valid', 'value': str(df["sex"].count())},
    #    {"type": 'invalid', 'value': str(df["sex"].isna().sum())}
    #]

    print("--- %s seconds ---" % (time.time() - start_time))

    #return df.describe()
    return {     
        'basic_sumstat': basic_sumstat,
        #'meta': meta,
        'describe': df.describe(percentiles=None, include='all').to_dict(),
        #'dfcount': df["sex"].value_counts().to_dict(),
        'time': "--- %s seconds ---" % (time.time() - start_time)
    }


    


@app.post("/upload/{item_id}")
async def upload_file(item_id: int):    
    datautils= DataUtils()
    return {"item_id": item_id, "file_contents": datautils.importFile(item_id)}
    return {"item_id + x": item_id}


def detect_column_types(df,meta):
    
    if meta.number_rows > 20000:
        df_sample=df.sample(n=5000, random_state=1)
        df_types=df_sample.convert_dtypes()
    else:        
        df_types=df.convert_dtypes()
    
    return df_types.dtypes.to_dict()
