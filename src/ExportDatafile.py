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
        
        return df, meta
            
        

    def export_file(self, params: DictParams):

        if (len(params.dtypes) == 0):
            dtypes=None
        else:
            dtypes=params.dtypes

        if (len(params.missings) == 0):
            params.missings=None
        
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
        variable_value_labels=self.parse_value_labels(variable_value_labels)

        # get all single character value labels that are not numeric and add them to missing_value_labels
        # this is to handle user-defined missing values in Stata and SPSS
        # e.g. .a, .b, .c etc.        
        params.missings=self.combine_missing_values_with_value_labels(variable_value_labels, params.missings)

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


        if params.export_format == 'csv':
            df.to_csv(output_file_path, index=False)
        elif params.export_format in ['dta','stata']:
            pyreadstat.write_dta(df, output_file_path, missing_user_values=params.missings, variable_value_labels=variable_value_labels, column_labels=params.name_labels)
        elif params.export_format in ['spss','sav']:
            pyreadstat.write_sav(df, output_file_path, missing_ranges=params.missings, variable_value_labels=variable_value_labels, column_labels=params.name_labels)
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
        - If any data in the column is a float, convert all numeric values to float

        Parameters
        ----------
        series : pd.Series
            A pandas Series with object type and mixed content.
            
        Returns
        -------
        pd.Series
            A new Series with numeric strings converted to int, others unchanged.
        """

        contains_float = series.apply(lambda x: isinstance(pd.to_numeric(x, errors='coerce'), float)).any()

        def try_convert(x):
            try:
                # If there's any float, convert all numeric values to float
                if contains_float:
                    return float(x) if isinstance(x, (str, int, float)) and pd.to_numeric(x, errors='coerce') is not None else x
                # Otherwise, convert numeric strings to int
                elif isinstance(x, str) and x.isdigit():
                    return int(x)
                else:
                    return x  # Non-numeric strings remain unchanged
            except (ValueError, TypeError):
                print(f"Could not convert {x} to numeric, keeping as is")
                return x

        return series.map(try_convert)
    

    def combine_missing_values_with_value_labels(self, variable_value_labels, missing_values):
        """
        Update the missing_values dictionary with single character value labels
        from variable_value_labels.        
        e.g. .a, .b, .c etc.
        
        Parameters
        ----------
        variable_value_labels : dict
            Dictionary of variable value labels.
        missing_values : dict
            Dictionary of user-defined missing values.
        
        Returns
        -------
        dict
            Updated missing_values dictionary with combined values.
        """
        
        if variable_value_labels is None:
            return missing_values
        
        combined_missing_values = missing_values.copy() if missing_values else {}                
        for var, labels in variable_value_labels.items():
            # Check if any label is a single character and not numeric
            for key, value in labels.items():
                # check if key is between 'a' and 'z'
                if isinstance(key, str) and len(key) == 1 and key.isalpha():
                    # If it's a single character, add it to missing_values
                    if var not in combined_missing_values:
                        combined_missing_values[var] = []
                    combined_missing_values[var].append(key)

        return combined_missing_values


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
        


    
