# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 17:12:00 2017

@author: mark
"""

import numpy as np
import pandas as pd
import statistics
import numpy.linalg as ln
import scipy.stats
import scipy.fftpack
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def extract_feature(time_series):
    data_mean = statistics.mean(time_series)
    data_std = statistics.stdev(time_series)
    data_rms = np.sqrt(sum(np.square(time_series))/len(time_series))
    data_kurtosis = scipy.stats.kurtosis(time_series)+3
    data_CF = max(np.abs(time_series))/data_rms
    data_skewness = scipy.stats.skew(time_series)
    data_ptp = max(time_series)-min(time_series)
    transform_record = np.array([data_mean, data_std, data_rms, data_kurtosis, 
                                 data_CF, data_skewness, data_ptp])
    return transform_record

def transform_time_window(time_series, window_size, offset):
    data = []
    for i in range(0, len(time_series), offset):
        if len(time_series[i:i+window_size]) < window_size:
            break
        transform_record = extract_feature(
                time_series[i:i+window_size].values.tolist())
        data.append(transform_record)
    return pd.DataFrame(data)

def get_pca_component(eign_values, percent):
    for i in range(len(eign_values)):
        if sum(eign_values[0:i+1])/sum(eign_values) > percent:
            main_component_index = i
            return main_component_index

# train data
train_data = pd.read_csv("fan_data/fan1.csv")
train_series = pd.Series(train_data['OLACTIVEPOWER'].values, 
                         name="OLACTIVEPOWER", index=train_data['REPORTTIME'])
# 選擇弱風速
train_series = train_series[9:389]
train_data_window = transform_time_window(train_series, 3, 1)

# test data
test_data = pd.read_csv("fan_data/fan1_error.csv")
test_series = pd.Series(test_data['OLACTIVEPOWER'].values, 
                         name="OLACTIVEPOWER", 
                         index=pd.to_datetime(test_data['REPORTTIME']))
# reverse
test_series = test_series[::-1]
test_experiment_series = test_series['2016-10-06 14:24':'2016-10-06 15:35']
test_data_window = transform_time_window(test_series, 3, 1)

# 標準化
scaler = StandardScaler()
scaler.fit(train_data_window)
scale_train_window = scaler.transform(train_data_window)
scale_test_window = scaler.transform(test_data_window)

# PCA compute
cov_matrix=np.cov(scale_train_window.T)  
eig_value, eig_vectors=ln.eig(cov_matrix)

# 取得 95% 重要的前幾個主成分
main_component_index = get_pca_component(eig_value, 0.95)

eiv_diag_matrix=np.matrix(ln.inv(np.diag(eig_value[0:main_component_index+1])))

anomaly_score_list = []
scale_test_window = np.matrix(scale_test_window)
eig_vectors = np.matrix(eig_vectors)
eig_value=np.matrix(eig_value)
tp_m = np.transpose(scale_test_window)
tp_eiv=np.transpose(eig_vectors)

for num in range(len(scale_test_window)):
    # can not understand
    anomaly_score = scale_test_window[num]*tp_eiv[:,0:main_component_index+1]*eiv_diag_matrix*eig_vectors[0:main_component_index+1,:]*tp_m[:,num]
    anomaly_score = anomaly_score.item()
    anomaly_score_list.append(anomaly_score)
anomaly_score_list = np.array(anomaly_score_list)
anomaly_score_list = anomaly_score_list/(max(anomaly_score_list)-min(anomaly_score_list))     

plt.figure(1)
plt.subplot(211)
plt.plot(test_series.values)
plt.subplot(212)
plt.plot(anomaly_score_list)
plt.show()
