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
import logging
import traceback

# Configure logging
logger = logging.getLogger(__name__)





class ExportDatafile:

    def load_file(self, fileinfo:FileInfo, usecols=None, dtypes=None):
        try:
            # Debug logging (only shown when LOG_LEVEL=DEBUG)
            logger.debug(f"Loading file: {fileinfo.file_path}, usecols: {usecols}, dtypes: {dtypes}")
            
            file_ext=os.path.splitext(fileinfo.file_path)[1]

            if file_ext.lower() == '.dta':
                try:
                    logger.debug(f"Reading DTA file: {fileinfo.file_path}")
                    df,meta = pyreadstat.read_dta(fileinfo.file_path, usecols=usecols)
                    logger.debug(f"DTA file loaded successfully, shape: {df.shape}")
                except UnicodeDecodeError as e:
                    logger.debug(f"DTA file Unicode decode error, trying with latin1 encoding: {e}")
                    df,meta = pyreadstat.read_dta(fileinfo.file_path, usecols=usecols, encoding="latin1")
                    logger.debug(f"DTA file loaded with latin1 encoding, shape: {df.shape}")

            elif file_ext.lower() == '.sav':
                logger.debug(f"Reading SAV file: {fileinfo.file_path}")
                df, meta = pyreadstat.read_sav(fileinfo.file_path, usecols=usecols)   
                logger.debug(f"SAV file loaded successfully, shape: {df.shape}")
            elif file_ext.lower() == '.csv':
                logger.debug(f"Reading CSV file: {fileinfo.file_path}")
                encodings_to_try = [None, "utf-8", "latin1", "cp1252", "iso-8859-1", "cp850"]
                last_error = None
                
                for encoding in encodings_to_try:
                    try:
                        if encoding is None:
                            df = pd.read_csv(fileinfo.file_path, usecols=usecols, dtype=dtypes)
                        else:
                            df = pd.read_csv(fileinfo.file_path, usecols=usecols, dtype=dtypes, encoding=encoding)
                        
                        logger.debug(f"CSV file loaded successfully with encoding '{encoding}', shape: {df.shape}")
                        
                        meta = SimpleNamespace()
                        meta.column_names=df.columns.tolist()
                        meta.column_names_to_labels=dict()
                        meta.number_rows=df.shape[0]
                        meta.number_columns=df.shape[1]
                        meta.variable_value_labels=dict()
                        meta.dtypes=df.dtypes.to_dict()
                        
                        break
                        
                    except UnicodeDecodeError as e:
                        last_error = e
                        logger.debug(f"Failed to read CSV file with encoding '{encoding}': {str(e)}")
                        continue
                    except Exception as e:
                        last_error = e
                        logger.debug(f"Failed to read CSV file with encoding '{encoding}': {str(e)}")
                        continue
                
                if last_error and 'df' not in locals():
                    raise Exception(f"Failed to read CSV file with any encoding. Last error: {str(last_error)}")

            else:
                raise Exception(f"File format not supported: {file_ext}")
            
            logger.debug(f"File loaded successfully: {fileinfo.file_path}, shape: {df.shape}")
            return df, meta
            
        except Exception as e:
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "function": "load_file",
                "file_path": fileinfo.file_path,
                "usecols": usecols,
                "dtypes": dtypes
            }
            logger.error(f"Failed to load file: {error_info}")
            raise Exception(f"Failed to load file {fileinfo.file_path}: {str(e)}") from e
            
        

    def export_file(self, params: DictParams):
        try:
            # Debug logging (only shown when LOG_LEVEL=DEBUG)
            logger.debug(f"Starting export_file with params: {params}")
            
            if (len(params.dtypes) == 0):
                dtypes=None
            else:
                dtypes=params.dtypes

            if params.missings is None or len(params.missings) == 0:
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

            logger.debug(f"Loading file: {params.file_path}")
            df,meta = self.load_file(params,usecols=columns, dtypes=dtypes)
            variable_value_labels=self.parse_value_labels(variable_value_labels, params.export_format)

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
            # 
            # For SPSS: Skip conversion of object/string columns as SPSS supports string data types

            # For columns with user missings (e.g. .a, .b etc. in Stata)
            # try to convert to numeric
            if params.export_format not in ['csv']:
                for col in df.columns:
                    # only apply to columns in params.missings
                    #if params.missings and col in params.missings:
                    #    logger.debug(f"Converting mixed column to numeric: {col}")
                    #    #convert mixed columns to numeric
                    #    df[col] = self.convert_mixed_column(df[col])
                    #    continue
                    
                    # For SPSS export, skip conversion of object/string columns to preserve string data types
                    if params.export_format in ['spss', 'sav']:
                        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                            logger.debug(f"SPSS export: preserving string data type for column {col}")
                            continue
                    
                    # if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                    #     # test if column can be converted to numeric
                    #     try:
                    #         # Check if all non-null values can be converted to numeric
                    #         non_null_values = df[col].dropna()
                    #         if len(non_null_values) > 0:
                    #             # Try to convert to numeric
                    #             pd.to_numeric(non_null_values, errors='raise')
                    #             # If successful, convert the entire column
                    #             df[col] = pd.to_numeric(df[col], errors='coerce')
                    #             logger.debug(f"Converted column {col} to numeric")
                    #     except (ValueError, TypeError):
                    #         # Column cannot be converted to numeric, keep as is
                    #         logger.debug(f"Column {col} cannot be converted to numeric, keeping as is")
                    #         pass

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

            logger.debug(f"Exporting to format: {params.export_format}, output path: {output_file_path}")

            if params.export_format == 'csv':
                df.to_csv(output_file_path, index=False)
            elif params.export_format in ['dta','stata']:
                pyreadstat.write_dta(df, output_file_path, missing_user_values=params.missings, variable_value_labels=variable_value_labels, column_labels=params.name_labels)
            elif params.export_format in ['spss','sav']:
                # For SPSS export, convert special missing values to system missings
                # SPSS doesn't support special missing values like 'a', 'b', 'c'
                df_for_spss, spss_missing_ranges = self.prepare_data_for_spss_export(df, params.missings)
                
                # Validate that special missing values were properly converted
                if not self.validate_spss_export_data(df_for_spss, params.missings):
                    logger.warning("Special missing values detected in data prepared for SPSS export - this may cause issues")
                
                # Prepare value labels for SPSS export (numeric keys for numeric columns, all keys for string columns)
                spss_value_labels = self.prepare_value_labels_for_spss_export(variable_value_labels, df_for_spss)
                
                pyreadstat.write_sav(df_for_spss, output_file_path, missing_ranges=spss_missing_ranges, variable_value_labels=spss_value_labels, column_labels=params.name_labels)
            elif params.export_format == 'json':
                df.to_json(output_file_path, orient='records')
            elif params.export_format in ['sas','xpt']:
                pyreadstat.write_xport(df,output_file_path)
            else:
                raise Exception("file format not supported: " + params.export_format)

            logger.debug(f"Export completed successfully to: {output_file_path}")
            return {
                'status':'success',
                'output_file':output_file_path,
                'output_file_size': DataUtils.sizeof_fmt(os.path.getsize(output_file_path))
            }
            
        except Exception as e:
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "function": "export_file",
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
            logger.error(f"Export file failed: {error_info}")
            raise Exception(f"Export failed in export_file: {str(e)}") from e


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
        
        try:
            # Debug logging (only shown when LOG_LEVEL=DEBUG)
            logger.debug(f"combine_missing_values_with_value_labels called with: variable_value_labels={variable_value_labels}, missing_values={missing_values}")
            
            if variable_value_labels is None:
                logger.debug("variable_value_labels is None, returning missing_values as is")
                return missing_values
            
            # Fix: Handle case where missing_values is None
            if missing_values is None:
                logger.debug("missing_values is None, creating empty dict")
                combined_missing_values = {}
            else:
                logger.debug(f"Copying missing_values: {missing_values}")
                combined_missing_values = missing_values.copy()
                
            for var, labels in variable_value_labels.items():
                logger.debug(f"Processing variable: {var}, labels: {labels}")
                # Check if any label is a single character and not numeric
                for key, value in labels.items():
                    # check if key is between 'a' and 'z'
                    if isinstance(key, str) and len(key) == 1 and key.isalpha():
                        logger.debug(f"Found single character label: {key} for variable {var}")
                        # If it's a single character, add it to missing_values
                        if var not in combined_missing_values:
                            combined_missing_values[var] = []
                        combined_missing_values[var].append(key)

            logger.debug(f"Final combined missing values: {combined_missing_values}")
            return combined_missing_values
            
        except Exception as e:
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "function": "combine_missing_values_with_value_labels",
                "variable_value_labels": variable_value_labels,
                "missing_values": missing_values
            }
            logger.error(f"combine_missing_values_with_value_labels failed: {error_info}")
            raise Exception(f"Failed to combine missing values with value labels: {str(e)}") from e


    def parse_value_labels(self, value_labels, export_format=None):
        """convert values to numeric values, except for SPSS string columns"""
        try:
            # Debug logging (only shown when LOG_LEVEL=DEBUG)
            logger.debug(f"Parsing value labels: {value_labels}, export_format: {export_format}")
            
            output=dict()

            if value_labels is None:
                logger.debug("value_labels is None, returning None")
                return None

            for key, value in value_labels.items():
                logger.debug(f"Processing variable: {key}, labels: {value}")
                output[key]=dict()
                for k,v in value.items():                
                    # For SPSS export, preserve string keys as-is (they will be filtered later based on column type)
                    if export_format in ['spss', 'sav']:
                        output[key][k]=v
                        logger.debug(f"SPSS export: preserving key '{k}' as-is for variable {key}")
                    else:
                        # For other formats, convert string integers to numeric
                        if self.is_string_integer(k):
                            k=int(k)
                            logger.debug(f"Converted string key '{k}' to int for variable {key}")
                        #else:
                        #    raise ValueError(f"Categorical variable [{key}] has non-numeric category code [{k}]. Only numeric codes are supported.")
                        
                        output[key][k]=v

            logger.debug(f"Value labels parsed successfully: {output}")
            return output
            
        except Exception as e:
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "function": "parse_value_labels",
                "value_labels": value_labels
            }
            logger.error(f"Failed to parse value labels: {error_info}")
            raise Exception(f"Failed to parse value labels: {str(e)}") from e

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
        

    def prepare_data_for_spss_export(self, df, missing_values):
        """
        Prepare data for SPSS export by converting special missing values to system missings.
        SPSS doesn't support special missing values like 'a', 'b', 'c' - they must be converted to NaN.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to prepare for SPSS export.
        missing_values : dict
            Dictionary of missing values per variable.
            
        Returns
        -------
        tuple
            (df_for_spss, spss_missing_ranges) where df_for_spss has special missings converted to NaN
            and spss_missing_ranges only contains numeric missing values.
        """
        try:
            df_for_spss = df.copy()
            spss_missing_ranges = {}
            
            if missing_values:
                for var, missing_vals in missing_values.items():
                    if var in df_for_spss.columns:
                        # Convert special missing values to NaN
                        for missing_val in missing_vals:
                            if isinstance(missing_val, str) and missing_val.isalpha():
                                # Replace special missing values with NaN
                                df_for_spss[var] = df_for_spss[var].replace(missing_val, np.nan).infer_objects(copy=False)
                                logger.debug(f"Converted special missing value '{missing_val}' to NaN for variable '{var}' in SPSS export")
                        
                        # Only keep numeric missing values for SPSS missing_ranges
                        numeric_missings = [val for val in missing_vals if not (isinstance(val, str) and val.isalpha())]
                        if numeric_missings:
                            spss_missing_ranges[var] = numeric_missings
            
            logger.debug(f"SPSS export preparation - converted special missings to NaN, missing_ranges: {spss_missing_ranges}")
            return df_for_spss, spss_missing_ranges
            
        except Exception as e:
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "function": "prepare_data_for_spss_export",
                "missing_values": missing_values
            }
            logger.error(f"Failed to prepare data for SPSS export: {error_info}")
            raise Exception(f"Failed to prepare data for SPSS export: {str(e)}") from e
        

    def validate_spss_export_data(self, df, missing_values):
        """
        Validate that data is properly prepared for SPSS export.
        Ensures no special missing values (like 'a', 'b', 'c') remain in the data.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to validate for SPSS export.
        missing_values : dict
            Dictionary of missing values per variable.
            
        Returns
        -------
        bool
            True if data is valid for SPSS export, False otherwise.
        """
        try:
            if missing_values:
                for var, missing_vals in missing_values.items():
                    if var in df.columns:
                        # Check if any special missing values remain in the data
                        for missing_val in missing_vals:
                            if isinstance(missing_val, str) and missing_val.isalpha():
                                # Check if this special missing value still exists in the data
                                if df[var].isin([missing_val]).any():
                                    logger.warning(f"Special missing value '{missing_val}' still exists in variable '{var}' for SPSS export")
                                    return False
            
            logger.debug("SPSS export data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate SPSS export data: {str(e)}")
            return False
        

    def prepare_value_labels_for_spss_export(self, variable_value_labels, df):
        """
        Prepare value labels for SPSS export.
        For numeric columns, only numeric keys are supported.
        For string/object columns, both numeric and string keys are supported.
        
        Parameters
        ----------
        variable_value_labels : dict
            Dictionary of variable value labels.
        df : pd.DataFrame
            The dataframe to determine column data types.
            
        Returns
        -------
        dict
            Filtered value labels appropriate for each column's data type.
        """
        try:
            if variable_value_labels is None:
                logger.debug("variable_value_labels is None, returning None for SPSS export")
                return None
            
            spss_value_labels = {}
            
            for var, labels in variable_value_labels.items():
                spss_value_labels[var] = {}
                
                # Check if the column exists in the dataframe
                if var not in df.columns:
                    logger.debug(f"SPSS export: variable '{var}' not found in dataframe, skipping value labels")
                    continue
                
                # Determine if this is a string/object column
                is_string_column = (df[var].dtype == 'object' or pd.api.types.is_string_dtype(df[var]))
                
                for key, value in labels.items():
                    if is_string_column:
                        # For string/object columns, keep all value labels as-is (both numeric and string keys)
                        spss_value_labels[var][key] = value
                    else:
                        # For numeric columns, only include numeric keys
                        if isinstance(key, (int, float)) or (isinstance(key, str) and self.is_string_integer(key)):
                            # Convert string keys to numeric if needed
                            if isinstance(key, str):
                                numeric_key = int(key)
                            else:
                                numeric_key = key
                            
                            spss_value_labels[var][numeric_key] = value                            
                        else:
                            logger.debug(f"SPSS export: excluded non-numeric value label '{key}' -> '{value}' for numeric variable '{var}'")
            
            return spss_value_labels
            
        except Exception as e:
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "function": "prepare_value_labels_for_spss_export",
                "variable_value_labels": variable_value_labels
            }
            logger.error(f"Failed to prepare value labels for SPSS export: {error_info}")
            raise Exception(f"Failed to prepare value labels for SPSS export: {str(e)}") from e
        


    
