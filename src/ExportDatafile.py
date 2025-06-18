import json
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pydantic import BaseModel
import os
import pyreadstat
from src.FileInfo import FileInfo
from src.VarInfo import VarInfo
from src.DictParams import DictParams
from src.DataUtils import DataUtils
from statsmodels.stats.weightstats import DescrStatsW
from types import SimpleNamespace





class ExportDatafile:

    def load_file(self, fileinfo:FileInfo, usecols=None, dtypes=None):
        file_ext=os.path.splitext(fileinfo.file_path)[1]

        if file_ext.lower() == '.dta':
            try:
                df,meta = pyreadstat.read_dta(fileinfo.file_path, usecols=usecols)
            except UnicodeDecodeError as e:
                df,meta = pyreadstat.read_dta(fileinfo.file_path, usecols=usecols, encoding="latin1")

        elif file_ext.lower() == '.sav':
            df, meta = pyreadstat.read_sav(fileinfo.file_path, usecols=usecols)   
        elif file_ext.lower() == '.csv':
            df = pd.read_csv(fileinfo.file_path, usecols=usecols, dtype=dtypes)

            meta = SimpleNamespace()
            meta.column_names=df.columns.tolist()
            meta.column_names_to_labels=dict()
            meta.number_rows=df.shape[0]
            meta.number_columns=df.shape[1]
            meta.variable_value_labels=dict()
            meta.dtypes=df.dtypes.to_dict()
        else:
            raise Exception("file not supported" + file_ext)
            #return {"error": "file not supported" + file_ext}
        
        return df, meta
            
        

    def export_file(self, params: DictParams):

        if (len(params.dtypes) == 0):
            dtypes=None
        else:
            dtypes=params.dtypes

        if (len(params.missings) == 0):
            missing_ranges=None
        else:
            missing_ranges=params.missings

        #missing_ranges=None
        #dtypes=None

        #check if variable_value_labels exists
        if (len(params.value_labels) == 0):
            variable_value_labels=None
        else:
            variable_value_labels=params.value_labels

        #variable_value_labels=None

        if (len(params.var_names) == 0):
            columns=None
        else:
            columns=list(params.var_names)
            #include weights columns
            #for w in params.weights:
            #    columns.append(str(w.field))
            #    columns.append(str(w.weight_field))

        df,meta = self.load_file(params,usecols=columns, dtypes=dtypes)

        #df.fillna(pd.NA,inplace=True)        
        #df=df.convert_dtypes()        

        file_formats = {
            'csv': "csv",
            'stata': "dta",
            "dta": "dta",
            "sav": "sav",
            'spss': "sav",
            "sas": "xpt",
            "xpt": "xpt",
            "json": "json"
        }

        if params.export_format not in file_formats:
            raise Exception("file format not supported: " + params.export_format)
        
        output_folder_path = os.path.join(os.path.dirname(params.file_path),"tmp")
        output_file_path = os.path.join(output_folder_path,os.path.splitext(os.path.basename(params.file_path))[0] + '.' + file_formats[params.export_format])

        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        # check if user has write permissions to the output folder
        if not os.access(output_folder_path, os.W_OK):
            raise Exception(f"User does not have write permissions to the output folder: {output_folder_path}")

        variable_value_labels=self.parse_value_labels(variable_value_labels)

        if params.export_format == 'csv':
            df.to_csv(output_file_path, index=False)
        elif params.export_format in ['dta','stata']:
            pyreadstat.write_dta(df, output_file_path, missing_user_values=missing_ranges, variable_value_labels=variable_value_labels, column_labels=params.name_labels)
        elif params.export_format in ['spss','sav']:
            pyreadstat.write_sav(df, output_file_path, missing_ranges=missing_ranges, variable_value_labels=variable_value_labels, column_labels=params.name_labels)
        elif params.export_format == 'json':
            df.to_json(output_file_path, orient='records')
        elif params.export_format in ['sas','xpt']:
            pyreadstat.write_xport(df,output_file_path)
        else:
            raise Exception("file format not supported: " + params.export_format)

        return {
            'status':'success',
            'output_file':output_file_path,
            'output_file_size': DataUtils.sizeof_fmt(os.path.getsize(output_file_path))
        }


    def parse_value_labels(self, value_labels):
        """convert values to numeric values"""
        output=dict()

        if value_labels is None:
            return None

        for key, value in value_labels.items():
            output[key]=dict()
            for k,v in value.items():
                if self.is_string_integer(k):
                    k=int(k)
                else:
                    raise ValueError(f"Categorical variable [{key}] has non-numeric category code [{k}]. Only numeric codes are supported.")
                    
                output[key][k]=v

        return output

    def list_get_numeric_values(self, values):
        output=[]
        for value in values:
            try:
                output.append(int(value))
            except:
                pass
        
        return output

    def is_string_integer(self, value):
        """check if value is a string that can be converted to an integer"""
        try:
            int(value)
            return True
        except ValueError:
            return False
        


    
