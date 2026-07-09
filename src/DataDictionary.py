import json
import numpy as np
import pandas as pd
from pydantic import BaseModel
import os
import pyreadstat
import logging
from src.FileInfo import FileInfo
from src.VarInfo import VarInfo
from src.DictParams import DictParams
from src.DataUtils import DataUtils
from src.DataDictionaryWeightValidation import validate_weight_columns_for_descr_stats
from src.weighted_freq_key import weighted_freq_category_key, sort_category_items, merge_category_value_counts
from src.utils.dta_reader import read_dta, should_use_chunked_read, iter_dta_chunks, dta_read_snapshot
from src.utils.dta_chunked_stats import ChunkedDictionaryStats
from src.utils.stata_missing import replace_stata_extended_missings
from statsmodels.stats.weightstats import DescrStatsW
from fastapi.exceptions import HTTPException

logger = logging.getLogger(__name__)


def _missing_values_as_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


class DataDictionary:
    """ Generate data dictionary from a data file [Stata, SPSS] """

    def load_file(self, fileinfo:FileInfo, metadataonly=True, usecols=None):
        file_ext=os.path.splitext(fileinfo.file_path)[1]

        if file_ext.lower() == '.dta':
            try:
                df, meta = read_dta(
                    fileinfo.file_path,
                    metadataonly=metadataonly,
                    usecols=usecols,
                    user_missing=True,
                )
            except Exception as e:
                logger.error("Failed to read DTA file. Last error: %s", e)
                raise HTTPException(
                    400, detail=f"Failed to read DTA file. Last error: {str(e)}"
                ) from e

        elif file_ext.lower() == '.sav':
            encodings_to_try = [
                None, 
                "utf-8", 
                "latin1", 
                "cp1252", 
                "iso-8859-1", 
                "cp850",
                "cp437",
                "windows-1252",
                "ascii",
                "utf-16",
                "utf-32"
            ]
            
            df, meta = None, None
            last_error = None
                
            for encoding in encodings_to_try:
                try:
                    if encoding is None:
                        df, meta = pyreadstat.read_sav(
                            fileinfo.file_path, 
                            user_missing=True,
                            metadataonly=metadataonly,
                            usecols=usecols
                        )
                    else:
                        df, meta = pyreadstat.read_sav(
                            fileinfo.file_path, 
                            user_missing=True,
                            metadataonly=metadataonly,
                            usecols=usecols,
                            encoding=encoding
                        )
                    break  # Success, exit the loop
                    
                except (pyreadstat.ReadstatError, UnicodeDecodeError, UnicodeError, ValueError) as e:
                    last_error = e
                    continue  # Try next encoding
            
            # If all encodings failed, raise the last error
            if df is None or meta is None:
                # try all encodings again with user_missing=False                
                for encoding in encodings_to_try:
                    try:
                        if encoding is None:
                            df, meta = pyreadstat.read_sav(
                                fileinfo.file_path, 
                                user_missing=False,
                                metadataonly=metadataonly,
                                usecols=usecols
                            )
                        else:
                            df, meta = pyreadstat.read_sav(
                                fileinfo.file_path, 
                                user_missing=False,
                                metadataonly=metadataonly,
                                usecols=usecols,
                                encoding=encoding
                            )
                        break  # Success, exit the loop
                        
                    except (pyreadstat.ReadstatError, UnicodeDecodeError, UnicodeError, ValueError) as e:
                        last_error = e
                        continue  # Try next encoding
                
                # If still failed after second attempt
                if df is None or meta is None:
                    raise HTTPException(400, detail=f"Failed to read SAV file with any encoding (tried both user_missing=True and user_missing=False). Last error: {str(last_error)}")
        else:
            logger.error(f"File not supported: {file_ext}")
            raise HTTPException(400, detail="file not supported: " + file_ext)
        
        return df,meta


            
        
    def get_metadata(self, fileinfo: FileInfo):
        """Get basic metadata excluding summary statistics"""

        df,meta = self.load_file(fileinfo)
        variables=[]

        for name in meta.column_names:
            variables.append(
                {
                    'name':name,
                    'labl':meta.column_names_to_labels[name],
                    'var_intrvl': self.variable_measure(meta,name),
                    'var_format': self.variable_format(meta,name),
                    'var_catgry_labels': self.variable_categories(meta,name)
                }
            )

        basic_sumstat = {
            'rows':meta.number_rows,
            'columns':meta.number_columns,
            'variables':variables,        
        }

        return basic_sumstat
    


    def get_name_labels(
        self,
        fileinfo: FileInfo,
        expected_columns=None,
        include_file_info: bool = False,
        include_comparison: bool = False,
        columns_only: bool = False,
    ):
        """Get variable names/labels (and optionally file format info + column comparison)."""
        from src.utils.source_file_info import build_file_info, compare_columns

        df, meta = self.load_file(fileinfo)
        column_names = list(meta.column_names)

        if columns_only:
            result = {
                "rows": meta.number_rows,
                "columns": meta.number_columns,
                "column_names": column_names,
                "variables": [],
            }
        else:
            variables = []
            labels = getattr(meta, "column_names_to_labels", {}) or {}
            types = getattr(meta, "readstat_variable_types", {}) or {}
            for name in column_names:
                variables.append(
                    {
                        "name": name,
                        "labl": labels.get(name),
                        "var_format": types.get(name),
                    }
                )
            result = {
                "rows": meta.number_rows,
                "columns": meta.number_columns,
                "variables": variables,
            }

        if include_file_info:
            result["file_info"] = build_file_info(fileinfo.file_path, meta)

        if include_comparison and expected_columns is not None:
            result["comparison"] = compare_columns(column_names, expected_columns)

        return result
    

    def infer_column_types(self, df):
        """Infer column types for a dataframe"""

        obj_columns= df.select_dtypes('object').columns

        for col in obj_columns:
            df[col]=df[col].astype('category')

            try:
                df[col] = df[col].astype('Float64')
                df[col] = df[col].astype('Int64')
            except ValueError as e:
                logger.warning(f"Failed to convert column to numeric: {str(e)}")

        

    def get_data_dictionary(self, fileinfo: FileInfo):
        """Get data dictionary for a data file"""
        df,meta = self.load_file(fileinfo,metadataonly=False)
       
        df.fillna(pd.NA,inplace=True)
        df=df.convert_dtypes()

        variables = []
        for name in meta.column_names:
            variables.append(self.variable_summary(df,meta,name))
            
        return {
            'rows':meta.number_rows,
            'columns':meta.number_columns,
            'variables':variables
        }


    def _prepare_dataframe_missings(self, df, missings):
        """Apply user-missing replacement and numeric coercion used before summarization."""
        if missings:
            df.replace(missings, np.nan, inplace=True)

        for col in df.columns:
            user_missings = _missing_values_as_list((missings or {}).get(col))
            df[col] = replace_stata_extended_missings(df[col], user_missings)
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                try:
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        converted = pd.to_numeric(non_null_values, errors='coerce')
                        if converted.notna().sum() == len(non_null_values):
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    pass

        df.fillna(pd.NA, inplace=True)
        return df.convert_dtypes()

    def _variable_sumstats_from_column_stats(self, col_stats, user_missings=None):
        if col_stats.is_numeric:
            stddev = col_stats.stddev()
            return [
                {"type": "vald", "value": str(col_stats.valid_count)},
                {"type": "invd", "value": str(col_stats.invalid_count)},
                {"type": "min", "value": str(col_stats.min_value)},
                {"type": "max", "value": str(col_stats.max_value)},
                {"type": "mean", "value": str(col_stats.mean)},
                {"type": "stdev", "value": str(stddev if stddev is not None else "")},
            ]
        return [
            {"type": "vald", "value": str(col_stats.valid_count)},
            {"type": "invd", "value": str(col_stats.invalid_count)},
        ]

    def _variable_valid_range_from_column_stats(self, col_stats):
        if col_stats.is_numeric:
            return {
                "range": {
                    "UNITS": "REAL",
                    "count": int(col_stats.valid_count),
                    "min": str(col_stats.min_value),
                    "max": str(col_stats.max_value),
                }
            }
        return {
            "range": {
                "UNITS": "REAL",
                "count": int(col_stats.valid_count),
            }
        }

    def _variable_categories_from_column_stats(
        self,
        col_stats,
        meta,
        variable_name,
        user_missings=None,
        categorical_list=None,
    ):
        user_missings = user_missings or []
        categorical_list = categorical_list or []
        categories = {}

        if variable_name in meta.variable_value_labels:
            categories = meta.variable_value_labels[variable_name]
            categories_calc = col_stats.value_counts
        elif variable_name in categorical_list:
            categories_calc = col_stats.value_counts
        else:
            return []

        categories_calc = merge_category_value_counts(categories_calc)
        if variable_name in categorical_list and len(categories_calc) > 1000:
            sorted_items = sorted(
                categories_calc.items(), key=lambda item: item[1], reverse=True
            )[:1000]
            categories_calc = dict(sorted_items)

        output = []
        for cat, freq in sort_category_items(categories_calc.items()):
            is_missing = int(str(cat) in user_missings or cat in user_missings)
            catgry = {
                "value": str(cat),
                "stats": [{"type": "freq", "value": str(freq)}],
            }
            if is_missing:
                catgry["is_missing"] = 1
            output.append(catgry)

        if categories:
            is_numeric_column = col_stats.is_numeric
            for catgry in output:
                if is_numeric_column:
                    try:
                        catgry["labl"] = categories.get(int(catgry["value"]), "")
                    except (ValueError, TypeError):
                        catgry["labl"] = categories.get(catgry["value"], "")
                else:
                    catgry["labl"] = categories.get(catgry["value"], "")

        return output

    def _variable_summary_from_column_stats(
        self,
        col_stats,
        meta,
        variable_name,
        user_missings=None,
        categorical_list=None,
    ):
        user_missings = user_missings or []
        variable_categories = self._variable_categories_from_column_stats(
            col_stats,
            meta,
            variable_name,
            user_missings=user_missings,
            categorical_list=categorical_list,
        )
        variable_has_categories = bool(variable_categories)

        return {
            "name": variable_name,
            "labl": meta.column_names_to_labels[variable_name],
            "var_intrvl": self.variable_measure(
                meta, variable_name, variable_has_categories
            ),
            "loc_width": meta.variable_display_width[variable_name],
            "var_invalrng": {
                "values": self.variable_missing_values(meta, variable_name)
            },
            "var_valrng": self._variable_valid_range_from_column_stats(col_stats),
            "var_sumstat": self._variable_sumstats_from_column_stats(
                col_stats, user_missings=user_missings
            ),
            "var_catgry": variable_categories,
            "var_catgry_labels": self.variable_categories(meta, variable_name),
            "var_format": self.variable_format(meta, variable_name),
            "var_format_original": self.variable_format(meta, variable_name),
        }

    def _weighted_freq_from_stats(self, weighted_stats):
        output = {}
        for val, raw in weighted_stats.freq.items():
            k = weighted_freq_category_key(val)
            raw = float(raw)
            output[k] = int(round(raw)) if abs(raw - round(raw)) < 1e-9 else raw
        return output

    def _get_data_dictionary_variable_chunked(self, params: DictParams, columns):
        _, meta = read_dta(
            params.file_path,
            metadataonly=True,
            usecols=columns,
            user_missing=True,
        )

        if not params.missings or len(params.missings) == 0:
            if hasattr(meta, "missing_user_values") and meta.missing_user_values is not None:
                params.missings = meta.missing_user_values
            else:
                params.missings = {}

        missings_map = params.missings or {}
        stats = ChunkedDictionaryStats(list(meta.column_names))
        weight_pairs = [(str(w.field), str(w.weight_field)) for w in params.weights]
        validated_weights = not weight_pairs

        with dta_read_snapshot(params.file_path) as stable_path:
            for chunk, _chunk_meta in iter_dta_chunks(
                stable_path,
                usecols=columns,
                user_missing=True,
            ):
                chunk = self._prepare_dataframe_missings(chunk.copy(), missings_map)
                if weight_pairs and not validated_weights:
                    for weight in params.weights:
                        validate_weight_columns_for_descr_stats(
                            chunk, weight.field, weight.weight_field
                        )
                    validated_weights = True
                stats.update_chunk(chunk, missings_map, weight_pairs=weight_pairs)

        variables = []
        for name in meta.column_names:
            user_missings = _missing_values_as_list(missings_map.get(name, []))
            variables.append(
                self._variable_summary_from_column_stats(
                    stats.column_stats(name),
                    meta,
                    name,
                    user_missings=user_missings,
                    categorical_list=params.categorical,
                )
            )

        weights = {}
        if weight_pairs:
            for weight in params.weights:
                weighted_stats = stats.weighted_stats(str(weight.field))
                if weighted_stats is None:
                    continue
                weights[weight.field] = {
                    "wgt_freq": self._weighted_freq_from_stats(weighted_stats),
                    "wgt_mean": weighted_stats.mean(),
                    "wgt_stdev": weighted_stats.stddev(),
                }
            self.apply_weighted_freq_to_variables(variables, weights)

        return {
            "rows": meta.number_rows,
            "columns": meta.number_columns,
            "variables": variables,
            "weights": weights,
        }

    def get_data_dictionary_variable(self, params: DictParams):
        try:
            if (len(params.var_names) == 0):
                columns=None
            else:
                columns=list(params.var_names)
                #weights_list
                for w in params.weights:
                    columns.append(str(w.field))
                    columns.append(str(w.weight_field))

            file_ext = os.path.splitext(params.file_path)[1].lower()
            if file_ext == ".dta" and should_use_chunked_read(
                params.file_path,
                usecols=columns,
                user_missing=True,
            ):
                return self._get_data_dictionary_variable_chunked(params, columns)

            df,meta = self.load_file(params,metadataonly=False,usecols=columns)


            #get user missing values from meta
            # If params.missings is empty, get missing values from metadata
            if not params.missings or len(params.missings) == 0:
                if hasattr(meta, 'missing_user_values') and meta.missing_user_values is not None:
                    params.missings = meta.missing_user_values
                else:
                    params.missings = {}
            
            # Replace missing values with NaN if any are defined
            try:
                df = self._prepare_dataframe_missings(df, params.missings)
            except Exception as e:
                raise HTTPException(500, detail=f"Failed to process data types: {str(e)}") from e

            variables = []
            try:
                for name in meta.column_names:
                    user_missings = []
                    if params.missings:
                        for missing_col, missings in params.missings.items():
                            if missing_col == name:
                                user_missings = _missing_values_as_list(missings)
                                break
                    variables.append(
                        self.variable_summary(
                            df,
                            meta,
                            name,
                            user_missings=user_missings,
                            categorical_list=params.categorical,
                        )
                    )
            except Exception as e:
                logger.error(f"Failed to process variables: {str(e)}")
                raise HTTPException(500, detail=f"Failed to process variables: {str(e)}")

            weights = {}

            if len(params.weights) > 0:
                try:
                    missings_map = params.missings or {}
                    for weight in params.weights:
                        validate_weight_columns_for_descr_stats(
                            df, weight.field, weight.weight_field
                        )
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
                    self.apply_weighted_freq_to_variables(variables, weights)
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(500, detail=f"Failed to calculate weights: {str(e)}")
                
            
            return {
                'rows':meta.number_rows,
                'columns':meta.number_columns,
                'variables':variables,
                'weights':weights
                }
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            # Catch any other unexpected errors
            raise HTTPException(500, detail=f"Unexpected error in get_data_dictionary_variable: {str(e)}")


    def apply_weighted_freq_to_variables(self, variables, weights_obj):
        for variable in variables:
            if (variable['name'] in weights_obj):
                DataUtils.set_variable_wgt_mean(variable,weighted_mean=weights_obj[variable['name']]['wgt_mean'])
                DataUtils.set_variable_wgt_stddev(variable,value=weights_obj[variable['name']]['wgt_stdev'])
                for var_catgry in variable['var_catgry']:
                    var_catgry['stats'].append(
                        DataUtils.set_wgt_stats_by_value(
                            weights_obj,
                            field=variable['name'],
                            value=var_catgry['value'],
                        )
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
            k = weighted_freq_category_key(val)
            raw = float(result[val])
            output[k] = int(round(raw)) if abs(raw - round(raw)) < 1e-9 else raw

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
        


    def variable_decimal_percision(self, meta, variable_name):
        """Return the decimal percision for a variable in a dataframe"""

        return 0
        if meta.readstat_variable_types[variable_name] == 'double':
            return meta.original_variable_types[variable_name].split('.')[1].count('0')
        else:
            return 0

    def variable_measure(self, meta,variable_name,variable_has_categories=False):
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

        return measure_mappings[meta.variable_measure[variable_name]]

    def variable_valid_range(self, df,meta,variable_name,user_missings=list()):
        """Return a dictionary of summary statistics for a variable in a dataframe"""        
        
        if (len(user_missings) > 0):
            df[variable_name].replace(user_missings, np.nan, inplace=True)            

        summary_stats=df[variable_name].describe(percentiles=None)

        # Check if the column is numeric
        is_numeric_column = pd.api.types.is_numeric_dtype(df[variable_name])
        
        if is_numeric_column:
            # For numeric columns, return count, min, and max
            return {
                "range": {
                    "UNITS": "REAL",
                    "count": int(summary_stats.get('count',0)),
                    "min": str(summary_stats.get('min')),
                    "max": str(summary_stats.get('max'))
                }
            }
        else:
            # For non-numeric columns, return only count
            return {
                "range": {
                    "UNITS": "REAL",
                    "count": int(summary_stats.get('count',0))
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
            #convert missing values to numeric
            user_missings=self.list_get_numeric_values(user_missings)  
            df[variable_name].replace(user_missings, np.nan, inplace=True)

        summary_stats=df[variable_name].describe(percentiles=None)

        count_=df[variable_name].count()
        sum_=df[variable_name].isna().sum()

        # Check if the column is numeric
        is_numeric_column = pd.api.types.is_numeric_dtype(df[variable_name])
        
        if is_numeric_column:
            # For numeric columns, return all statistics
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
            # For non-numeric columns, return only vald and invd
            return [
                    {
                        "type": "vald",
                        "value": str(count_)
                    },
                    {
                        "type": "invd",
                        "value": str(sum_)
                    }
                ]

    def variable_format(self, meta,variable_name):

        variable_type=meta.readstat_variable_types[variable_name]

        output={
            "type": "unknown",
            "schema": "other",
            "readstat_type": variable_type,
            "data_format": meta.original_variable_types[variable_name],
            "is_date": self.is_date(meta, variable_name)
        }

        if variable_type == 'double' or variable_type == 'float' or variable_type[:3] == 'int':
            output["type"] = "numeric"
            output["schema"] = "other"
        elif variable_type == 'object' or variable_type == 'string':
            output["type"] = "character"
            output["schema"] = "other"
        else:
            output["original_type"] = variable_type
            output["schema"] = "other"
        
        return output



    def is_date(self, meta, variable_name):
        """Return True if the variable is a date (Stata, SPSS, or SAS)"""
        data_format_original = meta.original_variable_types.get(variable_name)

        if not isinstance(data_format_original, str):
            return False

        stata_date_formats = ("%td", "%tm", "%tq", "%tw", "%th", "%tc", "%ty")
        spss_date_prefixes = ("DATE", "ADATE", "EDATE", "SDATE", "JDATE", "DATETIME", "TIME", "DTIME", "WKDAY", "MONTH")
        sas_date_prefixes = ("DATE", "DATETIME", "YYMMDD", "MMDDYY", "DDMMYY", "TOD", "DT", "DTDATE", "TIME", "WEEKDATE", "WORDDATE")

        fmt = data_format_original.strip().upper().rstrip('.')

        # Check Stata (starts with %t)
        if data_format_original.startswith(stata_date_formats):
            return True
        # Check SPSS
        elif any(fmt.startswith(prefix) for prefix in spss_date_prefixes):
            return True
        # Check SAS
        elif any(fmt.startswith(prefix) for prefix in sas_date_prefixes):
            return True
        else:
            return False


    def variable_missing_values(self, meta,variable_name):
        """Return the missing values for a variable in a dataframe"""
        if variable_name in meta.missing_user_values:
            return meta.missing_user_values[variable_name]
        else:
            return []


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



    def variable_categories_calculated(
        self, df,meta,variable_name, max_freq=100, user_missings=list(), categorical_list=list()
    ):
        
        is_categorical=False
        categories=[]
        categories_calc=[]
        
        # Ensure user_missings is a list
        if not user_missings:
            user_missings = []
        
        # Check if variable exists in dataframe
        if variable_name not in df.columns:
            return []

        #get value counts [freq] by each unique value
        categories_calc=merge_category_value_counts(df[variable_name].value_counts())

        #check if meta field has value labels - if so, treat as categorical regardless of data type
        if (variable_name in meta.variable_value_labels):
            is_categorical=True    
            categories=meta.variable_value_labels[variable_name]
        else:
            # Strict opt-in when value labels are not present.
            if variable_name not in categorical_list:
                return []
            # Keep a guardrail for high-cardinality user-selected categorical fields.
            if len(categories_calc) > 1000:
                categories_calc = dict(
                    sorted(categories_calc.items(), key=lambda item: item[1], reverse=True)[:1000]
                )

        output=[]

        
        for cat,freq in sort_category_items(categories_calc.items()):

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

        #if labels are available add them    
        if (categories):
            # Check if the column is numeric to determine how to look up labels
            is_numeric_column = pd.api.types.is_numeric_dtype(df[variable_name])
            
            for catgry in output:
                if is_numeric_column:
                    try:
                        # For numeric columns, try to convert value to int first
                        catgry['labl']=categories.get(int(catgry['value']),'')
                    except (ValueError, TypeError):
                        # If conversion fails, use the value as-is
                        catgry['labl']=categories.get(catgry['value'],'')
                else:
                    # For non-numeric columns, use the value as-is
                    catgry['labl']=categories.get(catgry['value'],'')


        return output


    def variable_summary(self, df,meta,variable_name, user_missings=list(), categorical_list=list()):
        """Return a dictionary of summary statistics for a variable in a dataframe"""        
        variable_categories=self.variable_categories_calculated(
            df,meta,variable_name, user_missings=user_missings, categorical_list=categorical_list
        )
        variable_has_categories=False

        if (variable_categories):
            variable_has_categories=True
                

        return {
            "name": variable_name,
            "labl": meta.column_names_to_labels[variable_name],
            #"var_dcml": self.variable_decimal_percision(meta,variable_name),
            "var_intrvl": self.variable_measure(meta,variable_name,variable_has_categories),
            "loc_width": meta.variable_display_width[variable_name],
            #"missing_values": self.variable_missing_values(meta,variable_name),
            "var_invalrng":{
                "values": self.variable_missing_values(meta,variable_name)
            },
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
            "var_format": self.variable_format(meta,variable_name),
            "var_format_original": self.variable_format(meta,variable_name)           
                # {
                #   "type": "numeric",
                #   "schema": "other"
                #   }
            #"var_invalrng": {
            #    "values": []
            #}
        }
