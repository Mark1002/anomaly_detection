# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 16:30:01 2018

@author: mark
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_preprocess.time_data import TimeDataPreprocess
from sklearn.preprocessing import StandardScaler
from anomal_algo.pca import PCA
from sklearn.metrics import confusion_matrix

def remove_na_zero(df):
    df = df.dropna(axis=1)
    ts = df.any()
    df = df[ts[ts.values].index]
    return df

def normalize(vectors):
    vectors = np.array(vectors)
    return (vectors-vectors.min())/(vectors.max()-vectors.min())
    
def perform_threshold_predict(anomaly_score_series, threshold):
    predict_label = (anomaly_score_series > threshold).values
    return predict_label

df = pd.read_csv("CSV/light/light2017.csv")
df = remove_na_zero(df)

features = ["V", "A", "PF", "W"]
train_series = pd.Series(df[features[3]].values, name=features[3],
                         index=pd.to_datetime(df['REPORTTIME']))

train_series = train_series.sort_index()
# drop na
train_series[train_series == 0] = np.nan
train_series = train_series.dropna()
# remove outlier
train_series = TimeDataPreprocess.remove_outlier(train_series, 1)
# regular time series interval to 1 min
train_series_regular = train_series.resample('1T').bfill()
train_series_regular = train_series_regular.interpolate(method='time', 
                                                        limit_direction='both')
# random test series
normaly_list = list(train_series_regular['2017-12-27 00:00':'2017-12-27 18:31'])
abnormal_list = list(np.random.uniform(low=20, high=40, size=(1440-len(normaly_list),)))
test_series = pd.Series(normaly_list + abnormal_list,
                        name=features[3], 
                        index=pd.date_range('2017-12-27 00:00', 
                                            '2017-12-27 23:59', freq='1T'))

# shift to time window
train_df = TimeDataPreprocess.transform_time_window(train_series_regular, 10, 1)
test_df = TimeDataPreprocess.transform_time_window(test_series, 10, 1)
ground_true_label = np.array(
    [False for i in range(len(test_df[:'2017-12-27 18:31']))] + 
    [True for i in range(len(test_df['2017-12-27 18:32':]))]
)
# time series to supervised learning process
train_X = train_df.loc[:,(train_df.columns != 't')]
train_Y = train_df['t'].values

train_df_fe = (train_df.iloc[0:60000,]).apply(TimeDataPreprocess.extract_feature, 
               axis=1)
test_df_fe = test_df.apply(TimeDataPreprocess.extract_feature, axis=1)

scaler = StandardScaler()
scaler.fit(train_df_fe)
scale_train_window = scaler.transform(train_df_fe)
scale_test_window = scaler.transform(test_df_fe)

# PCA
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

anomaly_score_series = pd.Series(normalize(anomaly_score_list), name="recon_error", 
                                 index=test_df_fe.index)

fig, ax1 = plt.subplots(figsize=(10, 5))
x_axis = anomaly_score_series.index
y1 = test_series[9:].values
y2 = anomaly_score_series.values
line1 = ax1.plot(x_axis, y1, 'b-', label='W')
ax1.set_xlabel('Time')
ax1.set_ylabel('W', color='b')
ax1.tick_params('y', colors='b')
ax2 = ax1.twinx()
line2 = ax2.plot(x_axis, y2, 'r-', label='Anomaly score')
ax2.set_ylabel('Anomaly score', color='r')
ax2.tick_params('y', colors='r')
line3 = [ax2.axhline(y=0.1, color='g', linestyle='--', label='Threshold')]
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels)
fig.tight_layout()
plt.show()


predict_label = perform_threshold_predict(anomaly_score_series, 50)
con_matrix = confusion_matrix(ground_true_label, predict_label)
tn, fp, fn, tp = con_matrix.ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
