class DataDictionary:

    def variable_decimal_percision(self, meta, variable_name):
        """Return the decimal percision for a variable in a dataframe"""
        if meta.readstat_variable_types[variable_name] == 'double':
            return meta.original_variable_types[variable_name].split('.')[1].count('0')
        else:
            return 0

    def variable_measure(self, meta,variable_name):
        """Return the measure for a variable in a dataframe"""
        # var measure takes values: scale, ordinal, nominal or unknown

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

    def variable_valid_range(self, df,meta,variable_name):
        """Return a dictionary of summary statistics for a variable in a dataframe"""        
        
        summary_stats=df[variable_name].describe(percentiles=None)

        range_result={
            "range": {
            "UNITS": "REAL"
            }
        }

        stats_remove=['25%','50%','75%','top']
        for stats,value in summary_stats.items():
            if stats not in stats_remove:
                range_result['range'][stats]=str(value)

        return range_result

        return {
            "range": {
                "UNITS": "REAL",
                "count": str(summary_stats.get('count')),
                "min": str(summary_stats.get('min','')),
                "max": str(summary_stats.get('max','')),
                "mean": str(summary_stats.get('mean','')),
                "stdev": str(summary_stats.get('std',''))
            }
        }

    def variable_sumstats(self, df,meta,variable_name):
        summary_stats=df[variable_name].describe(percentiles=None)
        return [
                {
                    "type": "vald",
                    "value": str(df[variable_name].count())
                },
                {
                    "type": "invd",
                    "value": str(df[variable_name].isna().sum())
                },
                {
                    "type":"min",
                    "value": summary_stats.get('min')
                },
                {
                    "type":"max",
                    "value": summary_stats.get('max')
                },
                {
                    "type": "mean",
                    "value": summary_stats.get('mean'),
                },
                {
                    "type": "stdev",
                    "value": summary_stats.get('std')
                }
            ]

    def variable_format(self, meta,variable_name):

        variable_type=meta.readstat_variable_types[variable_name]

        if variable_type == 'double' or variable_type == 'float' or variable_type[:3] == 'int':
            return {
                "type": "numeric",
                "schema": "other"
            }
        elif variable_type == 'string':
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



    def variable_categories_calculated(self, df,meta,variable_name, max_freq=100):
        
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
            output.append({
                "value": str(cat),
                "labl": '',
                "stats": [
                    {
                    "type": "freq",
                    "value": str(freq)
                    }
                ]}
            )
        #if labels are available add them    
        if (categories):
            for catgry in output:
                catgry['labl']=categories.get(int(catgry['value']),'')
                #if (catgry['value'] in categories):
                #     if categories[catgry['value']]:
                #         catgry['labl']=categories[catgry['value']]


        return output


    def variable_summary(self, df,meta,variable_name):
        """Return a dictionary of summary statistics for a variable in a dataframe"""

        

        return {
            "name": variable_name,
            "labl": meta.column_names_to_labels[variable_name],
            "var_dcml": self.variable_decimal_percision(meta,variable_name),
            "var_intrvl": self.variable_measure(meta,variable_name),
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
            "var_sumstat": self.variable_sumstats(df,meta,variable_name),
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
            "var_catgry": self.variable_categories_calculated(df,meta,variable_name),
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
            "var_format": {
                "type": "numeric",
                "schema": "other"
            },
            "var_type": "numeric",
            "var_concept": [
                []
            ],
            "var_invalrng": {
                "values": []
            }
        }
