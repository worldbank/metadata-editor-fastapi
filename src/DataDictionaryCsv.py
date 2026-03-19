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
from src.DataDictionaryWeightValidation import validate_weight_columns_for_descr_stats
from statsmodels.stats.weightstats import DescrStatsW
from fastapi import HTTPException
from types import SimpleNamespace
import traceback
import logging

# Configure logging
logger = logging.getLogger(__name__)


def _missing_values_as_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _dtype_spec_declares_numeric(spec) -> bool:
    """True if client `dtypes` entry means the column should end up numeric."""
    if spec is None:
        return False
    if isinstance(spec, type):
        if spec is bool:
            return False
        return issubclass(spec, (int, float, np.integer, np.floating, np.number))
    if isinstance(spec, str):
        s = spec.strip()
        if s.lower() in ("bool", "boolean", "object", "string", "str", "category"):
            return False
        return any(
            s.startswith(p)
            for p in (
                "int",
                "Int",
                "uint",
                "UInt",
                "float",
                "Float",
                "double",
                "complex",
            )
        )
    return False


class DataDictionaryCsv:
    """Generate data dictionary from a csv file"""

    def load_file(self, fileinfo:FileInfo, metadataonly=True, usecols=None, dtypes=None):
        """Load a CSV file into a pandas dataframe return dataframe and metadata """
        try:
            logger.debug(f"Loading CSV file: {fileinfo.file_path}, usecols: {usecols}, dtypes: {dtypes}")
            
            file_ext=os.path.splitext(fileinfo.file_path)[1]

            if file_ext.lower() == '.csv':
                encodings_to_try = [None, "utf-8", "latin1", "cp1252", "iso-8859-1", "cp850"]
                last_error = None
                
                for encoding in encodings_to_try:
                    try:
                        if encoding is None:
                            df = pd.read_csv(fileinfo.file_path, usecols=usecols, dtype=dtypes)
                        else:
                            df = pd.read_csv(fileinfo.file_path, usecols=usecols, dtype=dtypes, encoding=encoding)
                        
                        meta = SimpleNamespace()
                        meta.column_names=df.columns.tolist()
                        meta.column_names_to_labels=dict()
                        meta.number_rows=df.shape[0]
                        meta.number_columns=df.shape[1]
                        meta.variable_value_labels=dict()
                        meta.dtypes=df.dtypes.to_dict()
                        
                        logger.debug(f"CSV file loaded successfully with encoding '{encoding}', shape: {df.shape}")
                        return df, meta
                        
                    except UnicodeDecodeError as e:
                        last_error = e
                        logger.debug(f"Failed to read CSV file with encoding '{encoding}': {str(e)}")
                        continue
                    except Exception as e:
                        last_error = e
                        logger.debug(f"Failed to read CSV file with encoding '{encoding}': {str(e)}")
                        continue
                
                if last_error:
                    raise Exception(f"Failed to read CSV file with any encoding. Last error: {str(last_error)}")
                else:
                    raise Exception("Failed to read CSV file with any encoding")
            else:
                logger.error(f"File format not supported: {file_ext}")
                return {"error": "file not supported" + file_ext}
                
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
            logger.error(f"Failed to load CSV file: {error_info}")
            raise Exception(f"Failed to load CSV file {fileinfo.file_path}: {str(e)}") from e
            
    @staticmethod
    def _dictionary_usecols(params: DictParams):
        """Column names to load: var_names plus weight field/weight_field, stable unique order."""
        if len(params.var_names) == 0:
            return None
        cols = list(params.var_names)
        for w in params.weights:
            cols.append(str(w.field))
            cols.append(str(w.weight_field))
        return list(dict.fromkeys(cols))

    def _numeric_intent_columns(self, params: DictParams) -> set:
        """Columns that we will coerce for numeric computations.

        Only columns explicitly declared numeric in `params.dtypes` are coerced.
        If a weight column isn't declared numeric and comes in as non-numeric
        (object/string), we intentionally fail later in `validate_weight_columns_for_descr_stats`.
        """
        names = set()
        for col, spec in (params.dtypes or {}).items():
            if _dtype_spec_declares_numeric(spec):
                names.add(str(col))
        return names

    def _csv_dtypes_for_read(self, params: DictParams, columns, numeric_intent: set):
        """
        Build dtype= for read_csv. Strict numeric specs on numeric-intent columns are deferred
        (read as str) so tokens that are not valid literals can be turned into NA by
        pd.to_numeric(..., errors='coerce') after params.missings replacement.
        """
        if columns is None:
            return None
        dtypes_in = params.dtypes or {}
        out = {}
        for c in columns:
            if c not in dtypes_in:
                continue
            spec = dtypes_in[c]
            if c in numeric_intent:
                out[c] = str
            else:
                out[c] = spec
        return out if out else None

    def _apply_numeric_intent_normalization(self, df: pd.DataFrame, numeric_intent: set):
        """After missings replacement: coerce numeric-intent columns; non-numeric tokens → NA."""
        for col in numeric_intent:
            if col not in df.columns:
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def get_data_dictionary_variable(self, params: DictParams):
        """
            Generate data dictionary from a CSV file

            Parameters
            ----------

            DictParams:
                file_path: str
                var_names: List = []
                weights: List[WeightsColumns] = []
                missings: dict={} #List[UserMissings] = []
                dtypes: dict = {}
                value_labels: dict = {}
                name_labels: dict = {}
                export_format: str = "csv"
        """
        try:
            logger.debug(f"Starting get_data_dictionary_variable for CSV file: {params}")
            columns = self._dictionary_usecols(params)
            numeric_intent = self._numeric_intent_columns(params)
            dtypes = self._csv_dtypes_for_read(params, columns, numeric_intent)

            df, meta = self.load_file(params, metadataonly=False, usecols=columns, dtypes=dtypes)

            missings_map = params.missings or {}
            if missings_map:
                logger.debug(f"Replacing missing values: {missings_map}")
                df = df.replace(missings_map, np.nan)

            self._apply_numeric_intent_normalization(df, numeric_intent)

            variables = []
            for name in meta.column_names:
                user_missings = []
                if missings_map:
                    for missing_col, missings in missings_map.items():
                        if missing_col == name:
                            user_missings = _missing_values_as_list(missings)
                            break
                variables.append(
                    self.variable_summary(
                        df, meta, name, user_missings=user_missings, categorical_list=params.categorical
                    )
                )

            weights = {}

            if len(params.weights) > 0:
                for weight in params.weights:
                    validate_weight_columns_for_descr_stats(df, weight.field, weight.weight_field)
                    u_field = _missing_values_as_list(missings_map.get(weight.field))
                    u_wgt = _missing_values_as_list(missings_map.get(weight.weight_field))
                    weighted_ = self.calc_weighted_mean_n_stddev(
                        df,
                        weight.field,
                        weight.weight_field,
                        user_missings=u_field,
                        weight_missings=u_wgt,
                    )
                    weights[weight.field] = {
                        "wgt_freq": self.calc_weighted_freq(
                            df,
                            weight.field,
                            weight.weight_field,
                            user_missings=u_field,
                            weight_missings=u_wgt,
                        ),
                        "wgt_mean": weighted_["mean"],
                        "wgt_stdev": weighted_["stdev"],
                    }
                        
            #add weights stats to variables
            if weights:
                self.apply_weighted_freq_to_variables(variables, weights)
                            
            result = {
                'rows':meta.number_rows,
                'columns':meta.number_columns,
                'variables':variables,
                'weights':weights
            }
            return result

        except HTTPException:
            raise
        except Exception as e:
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "function": "get_data_dictionary_variable",
                "file_path": params.file_path,
                "var_names": params.var_names,
                "weights": params.weights,
                "missings": params.missings,
                "dtypes": params.dtypes
            }
            logger.error(f"ERROR in get_data_dictionary_variable: {error_info}")
            raise Exception("ERROR in get_data_dictionary_variable: " + str(e)) from e


    def apply_weighted_freq_to_variables(self, variables, weights_obj):
        for variable in variables:
            if (variable['name'] in weights_obj):
                DataUtils.set_variable_wgt_mean(variable,weighted_mean=weights_obj[variable['name']]['wgt_mean'])
                DataUtils.set_variable_wgt_stddev(variable,value=weights_obj[variable['name']]['wgt_stdev'])
                for var_catgry in variable['var_catgry']:
                    #check for is_missing
                    if ('is_missing' in var_catgry):
                        logger.warning(f"Skipping variable with category is_missing: {var_catgry}")
                        raise Exception("is_missing not supported")
                    var_catgry['stats'].append(
                        DataUtils.set_wgt_stats_by_value(weights_obj,field=variable['name'],value=int(var_catgry['value']))
                    )




    def calc_weighted_freq(
        self, df, col_name, wgt_col_name, user_missings=None, weight_missings=None
    ):
        new = df[[col_name, wgt_col_name]].copy()
        u_field = [] if user_missings is None else list(user_missings)
        u_wgt = [] if weight_missings is None else list(weight_missings)
        if u_field:
            new[col_name] = new[col_name].replace(u_field, np.nan)
        if u_wgt:
            new[wgt_col_name] = new[wgt_col_name].replace(u_wgt, np.nan)
        new.dropna(inplace=True)
        result = new.groupby(col_name)[wgt_col_name].sum().to_dict()

        output = {}
        for val in result:
            output[int(val)] = int(result[val])

        return output

    def calc_weighted_mean(
        self, df, col_name, wgt_col_name, user_missings=None, weight_missings=None
    ):
        new = df[[col_name, wgt_col_name]].copy()
        u_field = [] if user_missings is None else list(user_missings)
        u_wgt = [] if weight_missings is None else list(weight_missings)
        if u_field:
            new[col_name] = new[col_name].replace(u_field, np.nan)
        if u_wgt:
            new[wgt_col_name] = new[wgt_col_name].replace(u_wgt, np.nan)
        new.dropna(inplace=True)

        wdf = DescrStatsW(new[col_name], new[wgt_col_name], ddof=1)
        return wdf.mean

    def calc_weighted_mean_n_stddev(
        self, df, col_name, wgt_col_name, user_missings=None, weight_missings=None
    ):
        new = df[[col_name, wgt_col_name]].copy()
        u_field = [] if user_missings is None else list(user_missings)
        u_wgt = [] if weight_missings is None else list(weight_missings)
        if u_field:
            new[col_name] = new[col_name].replace(u_field, np.nan)
        if u_wgt:
            new[wgt_col_name] = new[wgt_col_name].replace(u_wgt, np.nan)
        new.dropna(inplace=True)

        wdf = DescrStatsW(new[col_name], new[wgt_col_name], ddof=1)
        return {"mean": wdf.mean, "stdev": wdf.std}
        
        
    
    #def calc_weighted_mean(self, df,col_name, wgt_col_name,user_missings=list()):
    #    wgt = df[col_name].replace(user_missings, np.nan)    
    #    return (wgt*df[wgt_col_name]).sum()/df[wgt_col_name].sum()
        
    

    def variable_measure(self, df, meta,variable_name,variable_has_categories=False):
        """Return the measure for a variable in a dataframe"""
        # var measure takes values: scale, ordinal, nominal or unknown

        if variable_has_categories:
            return 'discrete'

        value_labels = meta.variable_value_labels   
        
        if variable_name in value_labels:    
            return 'discrete'

        measure_mappings={
            'scale': 'contin',
            'ordinal': 'discrete',
            'nominal': 'discrete',
            'unknown': 'contin'
        }

        if is_numeric_dtype(df[variable_name]):
            return 'contin'
        elif is_string_dtype(df[variable_name]):
            return 'discrete'
        else:
            return measure_mappings['unknown']


    def variable_valid_range(self, df,meta,variable_name,user_missings=list()):
        """Return a dictionary of summary statistics for a variable in a dataframe"""        
        
        if (len(user_missings) > 0):
            # Use proper pandas method to avoid FutureWarning
            df[variable_name] = df[variable_name].replace(user_missings, np.nan)
                
        summary_stats=df[variable_name].describe(percentiles=None)

        return {
            "range": {
                "UNITS": "REAL",
                "count": int(summary_stats.get('count',0)),
                "min": str(summary_stats.get('min')),
                "max": str(summary_stats.get('max'))
            }
        }

    def list_get_numeric_values(self, values):
        output=[]
        for value in values:
            try:
                output.append(int(value))
            except:
                pass
        
        return output
        



    def variable_sumstats(self, df,meta,variable_name, user_missings=list()): 
        if (len(user_missings) > 0):
            user_missings=self.list_get_numeric_values(user_missings)
            # Use proper pandas method to avoid FutureWarning
            df[variable_name] = df[variable_name].replace(user_missings, np.nan)
            logger.debug(f"Applied user missing values to variable {variable_name}: {user_missings}")

        summary_stats=df[variable_name].describe(percentiles=None)

        count_=df[variable_name].count()
        sum_=df[variable_name].isna().sum()

        # Check if column is numeric to determine which stats to include
        is_numeric = is_numeric_dtype(df[variable_name])
        
        if is_numeric:
            # For numeric columns, include all statistics
            return [
                {
                    "type": "vald",
                    "value": str(count_)
                },
                {
                    "type": "invd",
                    "value": str(sum_)
                },
                {
                    "type":"min",
                    "value": str(summary_stats.get('min')) 
                },
                {
                    "type":"max",
                    "value": str(summary_stats.get('max'))
                },
                {
                    "type": "mean",
                    "value": str(summary_stats.get('mean'))
                },
                {
                    "type": "stdev",
                    "value": str(summary_stats.get('std'))
                }
            ]
        else:
            # For string columns, only include basic counts and frequency stats
            return [
                {
                    "type": "vald",
                    "value": str(count_)
                },
                {
                    "type": "invd",
                    "value": str(sum_)
                },
                {
                    "type": "unique",
                    "value": str(summary_stats.get('unique', 'N/A'))
                },
                {
                    "type": "top",
                    "value": str(summary_stats.get('top', 'N/A'))
                },
                {
                    "type": "freq",
                    "value": str(summary_stats.get('freq', 'N/A'))
                }
            ]

    def variable_format(self, df,meta,variable_name):
        """
        Return variable format as numeric or character
        """

        if is_numeric_dtype(df[variable_name]):
            return {
                "type": "numeric",
                "dtype": str(np.array(df[variable_name].dtype)),
                "schema": "other"
                }
        elif is_string_dtype(df[variable_name]):
            return {
                "type": "character",
                "dtype": str(np.array(df[variable_name].dtype)),
                "schema": "other"
                }
        else:
            return {
                "type": "unknown",
                "original_type": str(meta.dtypes[variable_name]),
                "dtype": str(np.array(df[variable_name].dtype)),
                "schema": "other"
                }


    def variable_categories(self, meta,variable_name):
        
        value_labels = meta.variable_value_labels   
        var_catgry = []

        if variable_name in value_labels:
            for key, value in value_labels[variable_name].items():
                var_catgry.append({
                    "value": key,
                    "labl": value
                })            
        
        return var_catgry



    def variable_categories_calculated(self, df,meta,variable_name, max_freq=100, user_missings=list(), categorical_list=list()):
        is_categorical=False
        categories=[]
        categories_calc=[]
        numeric_columns=df.select_dtypes('int').columns

        # Check if variable is explicitly set as categorical by user
        is_user_categorical = variable_name in categorical_list

        # Skip processing if not numeric and not user-defined categorical
        if not is_user_categorical and variable_name not in numeric_columns:
            logger.debug(f"Variable {variable_name} not numeric and not user-defined categorical, skipping categorical calculation")
            return []

        #get value counts [freq] by each unique value
        categories_calc=df[variable_name].value_counts()

        #check if meta field has value labels
        if (variable_name in meta.variable_value_labels):
            is_categorical=True    
            categories=meta.variable_value_labels[variable_name]
            logger.debug(f"Variable {variable_name} has value labels, treating as categorical")
        elif is_user_categorical:
            # User-defined categorical variable - check if within reasonable limit
            if (categories_calc.count() > 1000):
                logger.warning(f"User-defined categorical variable {variable_name} has too many categories ({categories_calc.count()}), limiting to 1000")
                # Still process but limit the categories to top 1000 by frequency
                categories_calc = categories_calc.head(1000)
            is_categorical=True
            logger.debug(f"Variable {variable_name} is user-defined categorical with {categories_calc.count()} categories")
        else:
            #guess if variable is categorical
            #too many categories
            if (categories_calc.count() > max_freq):
                #not a categorical variable
                logger.debug(f"Variable {variable_name} has too many categories ({categories_calc.count()}), not categorical")
                return []

            # check value data type for non-integer values (only for non-user-defined categorical)
            for cat,freq in categories_calc.items():
                if (cat==''):
                    continue
                
                if (cat!=int(cat)):
                    logger.debug(f"Variable {variable_name} has non-integer values, not categorical")
                    return []

        output=[]

        
        for cat,freq in sorted(categories_calc.items()):

            is_missing=0
            if (str(cat) in user_missings):
                is_missing=1
            
            if (cat in user_missings):
                is_missing=1

            catgry={
                "value": str(cat),
                #"labl": '',
                "stats": [
                    {
                    "type": "freq",
                    "value": str(freq)
                    }
                ]}
        
            if (is_missing):
                catgry['is_missing']=1

            output.append(catgry)

        return output


    def variable_summary(self, df,meta,variable_name, user_missings=list(), categorical_list=list()):
        """Return a dictionary of summary statistics for a variable in a dataframe"""

        variable_categories=self.variable_categories_calculated(df,meta,variable_name, user_missings=user_missings, categorical_list=categorical_list)
        variable_has_categories=False

        if (variable_categories):
            variable_has_categories=True

        return {
            "name": variable_name,
            "labl": meta.column_names_to_labels.get(variable_name),
            "dtype":str(np.array(df[variable_name].dtype)),
            #"var_dcml": self.variable_decimal_percision(meta,variable_name),
            "var_intrvl": self.variable_measure(df, meta,variable_name,variable_has_categories),
            #"loc_width": meta.variable_display_width[variable_name],
            #"var_invalrng": {
            #    "values": [
            #    "9",
            #    "999"
            #        ]
            #    },
            "var_valrng": self.variable_valid_range(df,meta,variable_name),
            #{
            #    "range": {
            #        "UNITS": "REAL",
            #        "min": 0,
            #        "max": 3,
            #        "mean": 1.77761138150505,
            #        "stdev": 1.32420581252985
            #    }
            #},
            "var_sumstat": self.variable_sumstats(df,meta,variable_name,user_missings),
            #[
            #    {
            #    "type": "vald",
            #    "value": 2671
            #    },
            #    {
            #    "type": "invd",
            #    "value": 0
            #    }
            #],
            "var_catgry": variable_categories,
            "var_catgry_labels": self.variable_categories(meta,variable_name),
            #"var_catgry": [
            #    {
            #        "value": 0,
            #        "labl": "abc",
            #        "stats": [
            #            {
            #            "type": "freq",
            #            "value": 783
            #            }
            #        ]
            #    }
            #],
            "var_format": self.variable_format(df,meta,variable_name),
                # {
                #   "type": "numeric",
                #   "schema": "other"
                #   }
            #"var_invalrng": {
            #    "values": []
            #}
        }

    def is_categorical_column(self, df, meta, variable_name, max_unique_ratio=0.05, max_unique_count=50):
        """
        Determine if a column should be treated as categorical
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe containing the column
        meta : SimpleNamespace  
            Metadata object
        variable_name : str
            Name of the column to check
        max_unique_ratio : float
            Maximum ratio of unique values to total values (default: 0.05 = 5%)
        max_unique_count : int
            Maximum number of unique values to consider categorical (default: 50)
            
        Returns:
        --------
        bool : True if column should be treated as categorical
        """        
        
        # 1. Check if explicitly defined as categorical in metadata
        if variable_name in meta.variable_value_labels:
            logger.debug(f"Variable {variable_name} has value labels, treating as categorical")
            return True
            
        # 2. Get basic column info
        total_rows = len(df[variable_name])
        unique_count = df[variable_name].nunique()
        unique_ratio = unique_count / total_rows if total_rows > 0 else 0
        
        # 3. String/object columns with reasonable unique values are likely categorical
        if is_string_dtype(df[variable_name]) or df[variable_name].dtype == 'object':
            return unique_count <= max_unique_count
            
        # 4. For numeric columns, check if they have limited unique values
        if is_numeric_dtype(df[variable_name]):
            # Small number of unique values relative to total
            if unique_ratio <= max_unique_ratio and unique_count <= max_unique_count:
                return True
                
            # Integer columns with small range might be categorical (e.g., 1-5 ratings)
            if df[variable_name].dtype in ['int8', 'int16', 'int32', 'int64']:
                value_range = df[variable_name].max() - df[variable_name].min()
                if value_range <= 20 and unique_count <= max_unique_count:
                    return True
                    
        return False

    def get_categorical_info(self, df, meta, variable_name):
        """
        Get categorical information for a column
        
        Returns:
        --------
        dict : Contains is_categorical, unique_count, unique_ratio, and categories
        """
        total_rows = len(df[variable_name])
        unique_count = df[variable_name].nunique()
        unique_ratio = unique_count / total_rows if total_rows > 0 else 0
        
        is_categorical = self.is_categorical_column(df, meta, variable_name)
        
        categories = []
        if is_categorical:
            # Get value counts for categorical variables
            value_counts = df[variable_name].value_counts()
            categories = [{"value": str(val), "frequency": freq} 
                         for val, freq in value_counts.items()]
        
        return {
            "is_categorical": is_categorical,
            "unique_count": unique_count,
            "unique_ratio": unique_ratio,
            "total_rows": total_rows,
            "categories": categories
        }
            
