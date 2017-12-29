# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 15:17:03 2017

@author: mark
"""
import statistics
import numpy as np
import scipy.stats
import pandas as pd

class TimeDataPreprocess:
    @staticmethod
    def extract_feature(time_series):
        if len(time_series) > 1:
            data_mean = statistics.mean(time_series)
            data_std = statistics.stdev(time_series)
            data_rms = np.sqrt(sum(np.square(time_series))/len(time_series))
            data_kurtosis = scipy.stats.kurtosis(time_series)+3
            data_cf = max(np.abs(time_series))/data_rms
            data_skewness = scipy.stats.skew(time_series)
            data_ptp = max(time_series)-min(time_series)
            transform_record = np.array([data_mean, data_std, data_rms, data_kurtosis, 
                                     data_cf, data_skewness, data_ptp])
        else:
            transform_record = time_series
        return transform_record
    
    @staticmethod
    def transform_time_window(time_series, window_size, offset, is_extract_feature=False):
        data = []
        for i in range(0, len(time_series), offset):
            if len(time_series[i:i+window_size]) < window_size:
                break
            if is_extract_feature is False:
                transform_record = time_series[i:i+window_size].values.tolist()
            else:
                transform_record = TimeDataPreprocess.extract_feature(
                    time_series[i:i+window_size].values.tolist())
            data.append(transform_record)
        return pd.DataFrame(data)
    
    @staticmethod
    def remove_outlier(time_series):
        std = time_series.std()
        mean = time_series.mean()
        upper = mean + 2*std
        lower = mean - 2*std
        clear_series = time_series[(time_series > lower) & (time_series < upper)]
        return clear_series
