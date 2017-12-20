# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import statistics
import numpy
import numpy.fft
import numpy.linalg as ln
import scipy.stats
import scipy.fftpack
import math
import statsmodels.api as sm
import matplotlib.pyplot as plt
#from scipy.stats import rankdata
all_file=os.listdir("Training/Healthy")
file_nums=len(all_file)
health_matrix = []
for i in range(0,file_nums):
    target_file_name="Training/Healthy/" + all_file[i]
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
    data_fft = max(numpy.abs(scipy.fftpack.fft(target_file)/19200)[int(len(target_file)/2560*19):int(len(target_file)/2560*21)])
#   data_fft = numpy.abs(scipy.fftpack.fft(target_file)/19200)[285:315]
    target_data = list([data_mean,data_std,data_rms,data_kurtosis,data_CF,data_skewness,data_ptp,data_fft])
    health_matrix.append(target_data)
health_matrix=numpy.array(health_matrix)
health_matrix=numpy.transpose(health_matrix)        
all_file=os.listdir("Training/Faulty")
file_nums=len(all_file)
Faulty_matrix = []
for i in range(0,file_nums):
    target_file_name="Training/Faulty/" + all_file[i]
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
    data_fft = max(numpy.abs(scipy.fftpack.fft(target_file)/19200)[int(len(target_file)/2560*19):int(len(target_file)/2560*21)])
 #   data_fft = numpy.abs(scipy.fftpack.fft(target_file)/19200)[285:315]
    target_data = list([data_mean,data_std,data_rms,data_kurtosis,data_CF,data_skewness,data_ptp,data_fft])
    Faulty_matrix.append(target_data)
Faulty_matrix = numpy.array(Faulty_matrix)
Faulty_matrix = numpy.transpose(Faulty_matrix) 
all_file=os.listdir("Testing")
file_nums=len(all_file)
test_array = []
test_matrix = []
for i in range(0,file_nums):
    target_file_name="Testing/" + all_file[i]
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
    data_fft = max(numpy.abs(scipy.fftpack.fft(target_file)/19200)[int(len(target_file)/2560*19):int(len(target_file)/2560*21)])
    target_data = list([data_mean,data_std,data_rms,data_kurtosis,data_CF,data_skewness,data_ptp,data_fft])
    test_matrix.append(target_data)
    test_array.append(data_fft)
test_matrix = numpy.array(test_matrix)
test_matrix = numpy.transpose(test_matrix)
#fisher criteria   
fisher_table = []
for num_features in range(0,8):
    fc = math.pow(statistics.mean(health_matrix[num_features])-statistics.mean(Faulty_matrix[num_features]),2)/(statistics.variance(health_matrix[num_features])+statistics.variance(Faulty_matrix[num_features]))
    fisher_table.append(fc)
train_array=numpy.concatenate((health_matrix[7],Faulty_matrix[7]))
#logistic regression
cv_list=[]
for num in range(0,20):
    cv_list.append(0.95)
for num in range(20,40):
    cv_list.append(0.05)
result_array=numpy.concatenate((train_array,test_array))    
logit = sm.Logit(cv_list,train_array)
result=logit.fit()
plt.plot(result.predict(result_array))
plt.show()
#PCA start
PCA_test_matrix = []
norm_train_matrix = []
norm_health_matrix = []
for num in range(0,8):
    norm_train_matrix.append(numpy.concatenate((health_matrix[num],Faulty_matrix[num],test_matrix[num])))
for num in range(0,8):
    PCA_test_matrix.append((norm_train_matrix[num]-statistics.mean(health_matrix[num]))/statistics.stdev(health_matrix[num]))
    norm_health_matrix.append((health_matrix[num]-statistics.mean(health_matrix[num]))/statistics.stdev(health_matrix[num]))
cov_data=numpy.cov(norm_health_matrix)  
cov_eig,cov_eiv=ln.eig(cov_data)
rank_cov=cov_eig
#rank=rankdata(cov_eig)-1
#rank_list = []\\   
#for n1 in range(0,8):
#    for n2 in range(0,8):
#        if rank[n2] == 7-n1:
#           rank_list.append(n2) 
vn=0           
for n1 in range(0,8):
#    for n2 in range(0,n1+1):
    if sum(rank_cov[0:n1+1])/sum(rank_cov) >0.95:
        vn=n1
        break    
eiv_inv=numpy.matrix(ln.inv(numpy.diag(cov_eig[0:vn+1])))
PCA_T2 = []
PCA_test_matrix=numpy.matrix(PCA_test_matrix)
cov_eiv=numpy.matrix(cov_eiv)
cov_eig=numpy.matrix(cov_eig)
tp_m = numpy.transpose(PCA_test_matrix)
tp_eiv=numpy.transpose(cov_eiv)
for num in range(0,70):
#    PCA_T2.append(sum((sum((tp_m[num]*cov_eiv[0:vn+1])[0])*eiv_inv*cov_eiv[0:vn+1]*tp_m[num])[0]))
     PCA_T2.append((tp_m[num]*tp_eiv[:,0:vn+1]*eiv_inv*cov_eiv[0:vn+1,:]*PCA_test_matrix[:,num])[0][0,0])
PCA_T2=PCA_T2/(max(PCA_T2)-min(PCA_T2))     
plt.plot(PCA_T2)
plt.show()       
#train_cov=numpy.cov(train_array)   
