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

        #check if variable_value_labels exists
        if (len(params.value_labels) == 0):
            variable_value_labels=None
        else:
            variable_value_labels=params.value_labels

        if (len(params.var_names) == 0):
            columns=None
        else:
            columns=list(params.var_names)

        df,meta = self.load_file(params,usecols=columns, dtypes=dtypes)


        # Note: for exporting to STATA, SPSS
        #
        # Data with user-defined missing values (e.g. .a, .b etc) is stored as string in CSV files
        # and all numeric values e.g. 1, 2, 3 are stored as string such as '1', '2', '3'
        # To properly handle this, we need to convert these columns to numeric while preserving the user-defined missing values
        # otherwise, the numeric values will be treated as strings and stata or SPSS export will not recognize them as numeric.

        # For columns with user missings (e.g. .a, .b etc. in Stata)
        # try to convert to numeric
        for col in df.columns:
            # only apply to columns in params.missings
            if col in params.missings:
                #convert mixed columns to numeric
                df[col] = self.convert_mixed_column(df[col])
                #print (f"Converted mixed column: {col}", df[col].dtype)
                continue
            
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                # test if column can be converted to numeric
                try:
                    # Check if all non-null values can be converted to numeric
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        # Try converting to numeric - if successful with no NaN introduced, convert
                        converted = pd.to_numeric(non_null_values, errors='coerce')
                        # If no values became NaN during conversion, all values are numeric
                        if converted.notna().sum() == len(non_null_values):
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    # If any error occurs, leave column as is
                    pass

        
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


    def convert_mixed_column(self, series):
        """
        Convert a pandas Series with mixed string values:
        - Strings representing integers (positive or negative) are converted to int
        - Non-numeric strings remain unchanged

        Parameters
        ----------
        series : pd.Series
            A pandas Series with object type and mixed content.
            
        Returns
        -------
        pd.Series
            A new Series with numeric strings converted to int, others unchanged.
        """

        print("Converting mixed column with user missings", series.name)

        def try_convert(x):
            try:
                return int(x)
            except (ValueError, TypeError):
                print (f"Could not convert {x} to int, keeping as is")
                return x

        return series.apply(try_convert)

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
                #else:
                #    raise ValueError(f"Categorical variable [{key}] has non-numeric category code [{k}]. Only numeric codes are supported.")
                    
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
        


    
