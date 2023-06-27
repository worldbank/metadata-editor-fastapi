import json
import numpy as np
import pandas as pd
from pydantic import BaseModel
import os
import pyreadstat
from src.FileInfo import FileInfo
from src.VarInfo import VarInfo
from src.DictParams import DictParams
from src.DataUtils import DataUtils
from statsmodels.stats.weightstats import DescrStatsW




class DataDictionary:

    def load_file(self, fileinfo:FileInfo, metadataonly=True, usecols=None):
        file_ext=os.path.splitext(fileinfo.file_path)[1]
        #folder_path=os.path.dirname(fileinfo.file_path)
        #file_exists=os.path.exists(fileinfo.file_path)

        if file_ext.lower() == '.dta':
            df,meta = pyreadstat.read_dta(fileinfo.file_path, metadataonly=metadataonly, usecols=usecols, user_missing=True)
        elif file_ext.lower() == '.sav':
            df, meta = pyreadstat.read_sav(fileinfo.file_path, user_missing=True)
        else:
            return {"error": "file not supported" + file_ext}
        
        return df,meta
            
        
        


    # get basic metadata excluding summary statistics
    def get_metadata(self, fileinfo: FileInfo):
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
    




    # get name, label, format
    def get_name_labels(self, fileinfo: FileInfo):
        df,meta = self.load_file(fileinfo)
        variables=[]

        for name in meta.column_names:
            variables.append(
                {
                    'name':name,
                    'labl':meta.column_names_to_labels[name],
                    'var_format': meta.readstat_variable_types[name]
                }
            )

        basic_sumstat = {
            'rows':meta.number_rows,
            'columns':meta.number_columns,
            'variables':variables,        
        }

        return basic_sumstat
    

    def infer_column_types(self, df):
        """Infer column types for a dataframe"""
        obj_columns= df.select_dtypes('object').columns

        for col in obj_columns:
            df[col]=df[col].astype('category')

            try:
                df[col] = df[col].astype('Float64')
                df[col] = df[col].astype('Int64')
            except ValueError as e:
                print("failed to convert column to numeric" + str(e) )

        

    def get_data_dictionary(self, fileinfo: FileInfo):

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


    def get_data_dictionary_variable(self, params: DictParams):
        
        if (len(params.var_names) == 0):
            columns=None
        else:
            columns=list(params.var_names)
            #weights_list
            for w in params.weights:
                columns.append(str(w.field))
                columns.append(str(w.weight_field))

        #print ("columns: ",columns)
        df,meta = self.load_file(params,metadataonly=False,usecols=columns)

        df.fillna(pd.NA,inplace=True)
        #df.fillna(0,inplace=True)
        df=df.convert_dtypes()

        variables = []
        for name in meta.column_names:
            user_missings=[]
            for user_missing in params.missings:
                if user_missing.field == name:
                    user_missings=user_missing.missings
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

            print("weights: ",weights)           
        #add weights stats to variables
        self.apply_weighted_freq_to_variables(variables, weights)
            
        
        return {
            'rows':meta.number_rows,
            'columns':meta.number_columns,
            'variables':variables,
            'weights':weights
            }


    def apply_weighted_freq_to_variables(self, variables, weights_obj):
        for variable in variables:
            if (variable['name'] in weights_obj):
                DataUtils.set_variable_wgt_mean(variable,weighted_mean=weights_obj[variable['name']]['wgt_mean'])
                DataUtils.set_variable_wgt_stddev(variable,value=weights_obj[variable['name']]['wgt_stdev'])
                for var_catgry in variable['var_catgry']:            
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
        new.dropna(subset=[col_name], inplace=True)

        wdf=DescrStatsW(new[col_name],new[wgt_col_name], ddof=1)
        return {
            'mean': wdf.mean,
            'stdev': wdf.std
        }
        
        
    
    #def calc_weighted_mean(self, df,col_name, wgt_col_name,user_missings=list()):
    #    wgt = df[col_name].replace(user_missings, np.NaN)    
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
            df[variable_name].replace(user_missings, np.NaN, inplace=True)            

        summary_stats=df[variable_name].describe(percentiles=None)

        #summary_stats=df[variable_name].describe(percentiles=None)

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
            #convert missing values to numeric
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

    def variable_format(self, meta,variable_name):

        variable_type=meta.readstat_variable_types[variable_name]

        if variable_type == 'double' or variable_type == 'float' or variable_type[:3] == 'int':
            return {
                "type": "numeric",
                "schema": "other"
            }
        elif variable_type == 'object' or variable_type == 'string':
            return {
                "type": "character",
                "schema": "other"
            }
        else:
            return {
                "type": "unknown",
                "original_type": variable_type,
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
        
        is_categorical=False
        categories=[]
        categories_calc=[]
        numeric_columns=df.select_dtypes('int').columns

        if (variable_name not in numeric_columns):
            print ("variable not numeric", variable_name)
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
                

        return {
            "name": variable_name,
            "labl": meta.column_names_to_labels[variable_name],
            #"var_dcml": self.variable_decimal_percision(meta,variable_name),
            "var_intrvl": self.variable_measure(meta,variable_name,variable_has_categories),
            "loc_width": meta.variable_display_width[variable_name],
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
            "var_format": self.variable_format(meta,variable_name),
                # {
                #   "type": "numeric",
                #   "schema": "other"
                #   }
            #"var_invalrng": {
            #    "values": []
            #}
        }
