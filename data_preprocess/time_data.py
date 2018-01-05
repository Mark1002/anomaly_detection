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
            transform_record = pd.Series([data_mean, data_std, data_rms, 
                                          data_kurtosis, data_cf, 
                                          data_skewness, data_ptp], 
                                          index=['data_mean', 'data_std', 
                                                 'data_rms', 'data_kurtosis',
                                                 'data_cf', 'data_skewness',
                                                 'data_ptp'])
        else:
            transform_record = time_series
        return transform_record
    
    @staticmethod
    def transform_time_window(time_series, window_size, offset):
        df = pd.DataFrame()
        for i in range(window_size-1, 0, -1*offset):
            df['t-' + str(i)] = time_series.shift(i)
        df['t'] = time_series.values
        df = df.dropna()
        return df
    
    @staticmethod
    def remove_outlier(time_series, std_num):
        std = time_series.std()
        mean = time_series.mean()
        upper = mean + std_num * std
        lower = mean - std_num * std
        clear_series = time_series[(time_series > lower) & (time_series < upper)]
        return clear_series
