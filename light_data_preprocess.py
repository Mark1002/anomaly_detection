# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 16:30:01 2018

@author: mark
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_preprocess.time_data import TimeDataPreprocess

def remove_na_zero(df):
    df = df.dropna(axis=1)
    ts = df.any()
    df = df[ts[ts.values].index]
    return df

df = pd.read_csv("CSV/light/light2017.csv")
df = remove_na_zero(df)

features = ["V", "A", "PF", "W"]
train_series = pd.Series(df[features[3]].values, name=features[3],
                         index=pd.to_datetime(df['REPORTTIME']))
train_series[train_series == 0] = np.nan
train_series_regular = train_series.resample('1T').bfill()
train_series_regular.interpolate(method='time')
plt.plot(train_series)


ts = train_series['2017-04-18 12:00':'2017-04-18 15:00']
new_df = pd.DataFrame()
for i in range(3, 0, -1):
    new_df['t-' + str(i)] = ts.shift(i)
new_df['t'] = ts.values

slice_window = TimeDataPreprocess.transform_time_window(ts, 4, 1)

