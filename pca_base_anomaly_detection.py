# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:22:36 2017

@author: mark
"""
import os
import statistics
import numpy
import numpy.fft
import numpy.linalg as ln
import scipy.stats
import scipy.fftpack
import matplotlib.pyplot as plt


def extract_feature_from_dir(path):
    all_file=os.listdir(path)
    file_nums=len(all_file)
    matrix = []
    for i in range(0,file_nums):
        target_file_name=path + "/" + all_file[i]
        target_file = open(target_file_name,'r') 
        target_file = target_file.readlines()[5:]
        for j in range(0,len(target_file)):
            target_file[j] = target_file[j].strip()
            target_file[j] = float(target_file[j])  
        data_mean = statistics.mean(target_file)
        data_std = statistics.stdev(target_file)
        data_rms = numpy.sqrt(sum(numpy.square(target_file))/len(target_file))
        data_kurtosis = scipy.stats.kurtosis(target_file)+3
        data_CF = max(numpy.abs(target_file))/data_rms
        data_skewness = scipy.stats.skew(target_file)
        data_ptp = max(target_file)-min(target_file)
        # can not understand
        data_fft = max(numpy.abs(scipy.fftpack.fft(target_file)/19200)[int(len(target_file)/2560*19):int(len(target_file)/2560*21)])
        target_data = list([data_mean,data_std,data_rms,data_kurtosis,data_CF,data_skewness,data_ptp, data_fft])
        matrix.append(target_data)
    matrix=numpy.array(matrix)
    matrix=numpy.transpose(matrix)
    return matrix            

def get_pca_component(eign_values, percent):
    for i in range(len(eign_values)):
        if sum(eign_values[0:i+1])/sum(eign_values) > percent:
            main_component_index = i
            return main_component_index

health_matrix = extract_feature_from_dir("Training/Healthy")
Faulty_matrix = extract_feature_from_dir("Training/Faulty")
test_matrix = extract_feature_from_dir("Testing")

PCA_test_matrix = []
norm_total_matrix = []
norm_health_matrix = []

# normalize
for num in range(0,8):
    norm_total_matrix.append(numpy.concatenate((health_matrix[num],Faulty_matrix[num],test_matrix[num])))
for num in range(0,8):
    PCA_test_matrix.append((norm_total_matrix[num]-statistics.mean(health_matrix[num]))/statistics.stdev(health_matrix[num]))
    norm_health_matrix.append((health_matrix[num]-statistics.mean(health_matrix[num]))/statistics.stdev(health_matrix[num]))

# PCA compute
cov_data=numpy.cov(norm_health_matrix)  
eig_value, eig_vectors=ln.eig(cov_data)

vn = get_pca_component(eig_value, 0.95)

eiv_inv=numpy.matrix(ln.inv(numpy.diag(eig_value[0:vn+1])))

anomaly_score_list = []
PCA_test_matrix=numpy.matrix(PCA_test_matrix)
eig_vectors = numpy.matrix(eig_vectors)
eig_value=numpy.matrix(eig_value)
tp_m = numpy.transpose(PCA_test_matrix)
tp_eiv=numpy.transpose(eig_vectors)
for num in range(0,70):
    # can not understand
    anomaly_score = tp_m[num]*tp_eiv[:,0:vn+1]*eiv_inv*eig_vectors[0:vn+1,:]*PCA_test_matrix[:,num]
    anomaly_score = anomaly_score.item()
    anomaly_score_list.append(anomaly_score)
anomaly_score_list = numpy.array(anomaly_score_list)
anomaly_score_list = anomaly_score_list/(max(anomaly_score_list)-min(anomaly_score_list))     
plt.plot(anomaly_score_list)
plt.show()
