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
import traceback


class DataDictionaryCsv:
    """Generate data dictionary from a csv file"""

    def load_file(self, fileinfo:FileInfo, metadataonly=True, usecols=None, dtypes=None):
        """Load a CSV file into a pandas dataframe return dataframe and metadata """

        file_ext=os.path.splitext(fileinfo.file_path)[1]

        if file_ext.lower() == '.csv':
            df = pd.read_csv(fileinfo.file_path, usecols=usecols, dtype=dtypes)
            
            meta = SimpleNamespace()
            meta.column_names=df.columns.tolist()
            meta.column_names_to_labels=dict()
            meta.number_rows=df.shape[0]
            meta.number_columns=df.shape[1]
            meta.variable_value_labels=dict()
            meta.dtypes=df.dtypes.to_dict()
        else:
            return {"error": "file not supported" + file_ext}
        
        return df, meta
            
    

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

            if (len(params.dtypes) == 0):
                dtypes=None
            else:
                dtypes=params.dtypes

            if (len(params.var_names) == 0):
                columns=None
            else:
                columns=list(params.var_names)
                #weights_list
                for w in params.weights:
                    columns.append(str(w.field))
                    columns.append(str(w.weight_field))

            df,meta = self.load_file(params,metadataonly=False,usecols=columns, dtypes=dtypes)

            df.fillna(pd.NA,inplace=True)
            #df.fillna(0,inplace=True)
            df=df.convert_dtypes()

            variables = []
            for name in meta.column_names:
                user_missings=[]
                for missing_col, missings in params.missings.items():                
                    if missing_col == name:
                        user_missings=missings
                        break
                variables.append(self.variable_summary(df,meta,name,user_missings=user_missings))

            weights = {}

            if len(params.weights) > 0:
                for weight in params.weights:            
                    weighted_=self.calc_weighted_mean_n_stddev(df,weight.field, weight.weight_field)
                    weights[weight.field]={
                            'wgt_freq': self.calc_weighted_freq(df,weight.field, weight.weight_field),
                            'wgt_mean': weighted_['mean'],
                            'wgt_stdev': weighted_['stdev']
                        }
                        
            #add weights stats to variables
            self.apply_weighted_freq_to_variables(variables, weights)
                
            
            return {
                'rows':meta.number_rows,
                'columns':meta.number_columns,
                'variables':variables,
                'weights':weights
                }

        except Exception as e:
            print ("ERROR in get_data_dictionary_variable:", str(e))
            raise Exception("ERROR in get_data_dictionary_variable: " + str(e))


    def apply_weighted_freq_to_variables(self, variables, weights_obj):
        for variable in variables:
            if (variable['name'] in weights_obj):
                DataUtils.set_variable_wgt_mean(variable,weighted_mean=weights_obj[variable['name']]['wgt_mean'])
                DataUtils.set_variable_wgt_stddev(variable,value=weights_obj[variable['name']]['wgt_stdev'])
                for var_catgry in variable['var_catgry']:
                    #check for is_missing
                    if ('is_missing' in var_catgry):
                        raise Exception("is_missing not supported")
                        print ("skipping variable with category is_missing ", var_catgry)
                        continue
                    var_catgry['stats'].append(
                        DataUtils.set_wgt_stats_by_value(weights_obj,field=variable['name'],value=int(var_catgry['value']))
                    )




    def calc_weighted_freq(self, df, col_name, wgt_col_name):
        result=df.groupby(col_name)[wgt_col_name].sum().to_dict()

        output={}
        for val in result:
            output[int(val)]=int(result[val])

        return output

    
    def calc_weighted_mean(self, df,col_name, wgt_col_name,user_missings=list()):
        #create a copy of df
        new = df[[col_name,wgt_col_name]].copy()

        #replace user missings with NaN
        new[col_name]=df[col_name].replace(user_missings, np.NaN)

        #drop na values
        new.dropna(subset=[col_name], inplace=True)

        wdf=DescrStatsW(new[col_name],new[wgt_col_name], ddof=1)
        return wdf.mean
    

    
    def calc_weighted_mean_n_stddev(self, df,col_name, wgt_col_name,user_missings=list()):
        #create a copy of df
        new = df[[col_name,wgt_col_name]].copy()

        #replace user missings with NaN
        new[col_name]=df[col_name].replace(user_missings, np.NaN)

        #drop na values
        new.dropna(inplace=True)

        wdf=DescrStatsW(new[col_name],new[wgt_col_name], ddof=1)
        return {
            'mean': wdf.mean,
            'stdev': wdf.std
        }
        
        
    
    #def calc_weighted_mean(self, df,col_name, wgt_col_name,user_missings=list()):
    #    wgt = df[col_name].replace(user_missings, np.NaN)    
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
            df[variable_name].replace(user_missings, np.NaN, inplace=True)            
                
        summary_stats=df[variable_name].describe(percentiles=None)

        return {
            "range": {
                "UNITS": "REAL",
                "count": int(summary_stats.get('count',0)),
                "min": str(summary_stats.get('min')),
                "max": str(summary_stats.get('max')),
                #"mean": str(summary_stats.get('mean','')),
                #"stdev": str(summary_stats.get('std',''))
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

            df[variable_name].replace(user_missings, np.NaN, inplace=True)

        summary_stats=df[variable_name].describe(percentiles=None)

        count_=df[variable_name].count()
        sum_=df[variable_name].isna().sum()


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



    def variable_categories_calculated(self, df,meta,variable_name, max_freq=100, user_missings=list()):
        #print("user missings", user_missings)
        is_categorical=False
        categories=[]
        categories_calc=[]
        numeric_columns=df.select_dtypes('int').columns

        if (variable_name not in numeric_columns):
            #print ("variable not numeric, not categorical ", variable_name)
            return []

        #get value counts [freq] by each unique value
        categories_calc=df[variable_name].value_counts()

        #check if meta field has value labels
        if (variable_name in meta.variable_value_labels):
            is_categorical=True    
            categories=meta.variable_value_labels[variable_name]
        else:
            #guess if variable is categorical
            #too many categories
            if (categories_calc.count() > max_freq):
                #not a categorical variable
                return []

            # check value data type for non-integer values
            for cat,freq in categories_calc.items():
                if (cat==''):
                    continue
                
                if (cat!=int(cat)):
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

        #if labels are available add them    
        #if (categories):
        #    for catgry in output:
        #        catgry['labl']=categories.get(int(catgry['value']),'')

        return output


    def variable_summary(self, df,meta,variable_name, user_missings=list()):
        """Return a dictionary of summary statistics for a variable in a dataframe"""

        variable_categories=self.variable_categories_calculated(df,meta,variable_name, user_missings=user_missings)
        variable_has_categories=False

        if (variable_categories):
            variable_has_categories=True

        #print ("variable_categories",variable_name, variable_categories)        

        return {
            "name": variable_name,
            "labl": meta.column_names_to_labels.get(variable_name),
            "dtype":str(np.array(df[variable_name].dtype)),
            #"var_dcml": self.variable_decimal_percision(meta,variable_name),
            "var_intrvl": self.variable_measure(df, meta,variable_name,variable_has_categories),
            #"loc_width": meta.variable_display_width[variable_name],
            #TODO
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
