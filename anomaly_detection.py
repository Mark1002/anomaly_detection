# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 17:12:00 2017

@author: mark
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from data_preprocess.time_data import TimeDataPreprocess
from anomal_algo.pca import PCA

# train data
train_data = pd.read_csv("CSV/fan_data/fan1.csv")
train_series = pd.Series(train_data['OLACTIVEPOWER'].values, 
                         name="OLACTIVEPOWER", 
                         index=pd.to_datetime(train_data['REPORTTIME']))
# 選擇弱風速
train_series = train_series.sort_index()
train_series = train_series['2016-09-20 16:40':'2016-09-21 09:48']

# drop na
train_series[train_series == 0] = np.nan
train_series = train_series.dropna()

# regular time series interval to 3 min
train_series_regular = train_series.resample('3T').bfill()
train_series_regular = train_series_regular.interpolate(method='time', 
                                                        limit_direction='both')

# test data
test_data = pd.read_csv("CSV/fan_data/fan1_error.csv")
test_series = pd.Series(test_data['OLACTIVEPOWER'].values, 
                        name="OLACTIVEPOWER", 
                        index=pd.to_datetime(test_data['REPORTTIME']))
# reverse
test_series = test_series[::-1]
abnormal_series1 = list(test_series[:'2016-10-06 14:48'])
abnormal_series2 = list(test_series['2016-10-06 15:10':])
test_series = test_series['2016-10-06 14:24':'2016-10-06 15:35']
test_series = test_series.resample('3T').bfill()
# test_series = pd.Series(abnormal_series1 + list(np.random.uniform(50, 51, size=60)) + abnormal_series2)

# shift to time window
train_data_window = TimeDataPreprocess.transform_time_window(train_series, 10, 1)
train_data_window = train_data_window.apply(TimeDataPreprocess.extract_feature, axis=1)
test_data_window = TimeDataPreprocess.transform_time_window(test_series, 10, 1)
test_data_window = test_data_window.apply(TimeDataPreprocess.extract_feature, axis=1)

# 標準化
scaler = StandardScaler()
scaler.fit(train_data_window)
scale_train_window = scaler.transform(train_data_window)
scale_test_window = scaler.transform(test_data_window)

# PCA compute
pca = PCA()
pca.fit(scale_train_window)
pca_params = pca.get_params()
eig_vectors = pca_params['eig_vectors']
# 取得 95% 重要的前幾個主成分
components = pca.get_pca_component(0.95)
V = eig_vectors[:,0:components]

anomaly_score_list = []
for num in range(len(scale_test_window)):
    anomaly_score = pca.perform_reconstruction_error(scale_test_window[num], V)
    anomaly_score_list.append(anomaly_score)

anomaly_score_list = perform_normalize(anomaly_score_list)
  
plt.figure(1)
plt.subplot(211)
plt.title('anomaly_score')
plt.plot(anomaly_score_list)
plt.subplot(212)
plt.title('origin time series')
plt.plot(test_series.values[9:])
plt.tight_layout()
plt.show()
