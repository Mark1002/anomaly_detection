# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 17:12:00 2017

@author: mark
"""
import pandas as pd
import numpy as np
import numpy.linalg as ln
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from data_preprocess.time_data import TimeDataPreprocess

def get_pca_component(eign_values, percent):
    for i in range(len(eign_values)):
        if sum(eign_values[0:i+1])/sum(eign_values) > percent:
            return i + 1 

def perform_t_squared(X, V):
    components = V.shape[1]
    eiv_diag_matrix=np.matrix(ln.inv(np.diag(eig_value[0:components])))
    value = X*V*eiv_diag_matrix*V.T*X.T
    return value.item()
 
def perform_reconstruction_error(X, V):
    return ln.norm((X - X*V*V.T))

def perform_normalize(vectors):
    vectors = np.array(vectors)
    return vectors/(max(vectors)-min(vectors))

# train data
train_data = pd.read_csv("CSV/fan_data/fan1.csv")
train_series = pd.Series(train_data['OLACTIVEPOWER'].values, 
                         name="OLACTIVEPOWER", index=train_data['REPORTTIME'])
# 選擇弱風速
train_series = train_series[9:389]
train_data_window = TimeDataPreprocess.transform_time_window(train_series, 10, 1, True)

# test data
test_data = pd.read_csv("CSV/fan_data/fan1_error.csv")
test_series = pd.Series(test_data['OLACTIVEPOWER'].values, 
                        name="OLACTIVEPOWER", 
                        index=pd.to_datetime(test_data['REPORTTIME']))
# reverse
test_series = test_series[::-1]
abnormal_series1 = list(test_series[:'2016-10-06 14:48'])
abnormal_series2 = list(test_series['2016-10-06 15:10':])
test_experiment_series = test_series['2016-10-06 14:24':'2016-10-06 15:35']

test_series = pd.Series(abnormal_series1 + list(np.random.uniform(50, 51, size=60)) + abnormal_series2)
test_data_window = TimeDataPreprocess.transform_time_window(test_series, 10, 1, True)

# 標準化
scaler = StandardScaler()
scaler.fit(train_data_window)
scale_train_window = scaler.transform(train_data_window)
scale_test_window = scaler.transform(test_data_window)

# PCA compute
cov_matrix = np.cov(scale_train_window.T)  
eig_value, eig_vectors = ln.eig(cov_matrix)

# 取得 95% 重要的前幾個主成分
components = get_pca_component(eig_value, 0.95)

anomaly_score_list = []
anomaly_score_list2 = []
scale_test_window = np.matrix(scale_test_window)
eig_vectors = np.matrix(eig_vectors)

V = eig_vectors[:,0:components]

for num in range(len(scale_test_window)):
    # can not understand
    anomaly_score = perform_t_squared(scale_test_window[num], V)
    anomaly_score2 = perform_reconstruction_error(scale_test_window[num], V)
    anomaly_score_list.append(anomaly_score)
    anomaly_score_list2.append(anomaly_score2)
   

anomaly_score_list = perform_normalize(anomaly_score_list)
anomaly_score_list2 = perform_normalize(anomaly_score_list2)
     
plt.figure(1)
plt.subplot(411)
plt.title('origin time series')
plt.plot(test_series.values)
plt.subplot(413)
plt.title('anomaly_score')
plt.plot(anomaly_score_list)
plt.subplot(414)
plt.plot(anomaly_score_list2)
plt.show()
