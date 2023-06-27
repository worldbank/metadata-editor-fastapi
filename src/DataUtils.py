class DataUtils:
            
    def sizeof_fmt(num, suffix="B"):
        for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
            if abs(num) < 1024.0:
                return f"{num:3.1f}{unit}{suffix}"
            num /= 1024.0
        return f"{num:.1f}Yi{suffix}"
    

    #
    def set_wgt_stats_by_value( weightsDict,field,value):
        
        #print ("weightDict", weightsDict, field, value)

        if (field not in weightsDict):
            print ("field not found", field)
            return {}
        
        freq_values = weightsDict[field]['wgt_freq']

        result={
            "type": "freq",
            "wgtd": "wgtd",
            "value": freq_values[value]
        }
        
        return result

        

    def set_wgt_stats_by_valuex( weightsDict,field,value):
        if (field not in weightsDict):
            print ("field not found", field)
            return {}
        
        freq_values = list(weightsDict[field]['wgt_freq']['value'].values())

        if (value not in freq_values):
            print ("value not found", value)
            print ("freq_values", freq_values)
            return {}
        
        
        idx=freq_values.index(value)
        freq_counts = list(weightsDict[field]['wgt_freq']['wghtd'].values())

        result={
            "type": "freq",
            "wgtd": "wgtd",
            "value": freq_counts[idx]
        }
        
        return result
    
    def set_variable_wgt_mean(variable,weighted_mean):
        
        variable['var_sumstat'].append({
            "type": "mean",
            "wgtd": "wgtd",
            "value": weighted_mean
        })

    def set_variable_wgt_stddev(variable,value):
        
        variable['var_sumstat'].append({
            "type": "stdev",
            "wgtd": "wgtd",
            "value": value
        })

