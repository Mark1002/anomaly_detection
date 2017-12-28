# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 15:02:24 2017

@author: mark
"""
import numpy as np
import numpy.linalg as ln

class PCA:
    def __init__(self):
        self.eig_values = None
        self.eig_vectors = None
        self.is_fit = False
        
    def fit(self, X):
        cov_matrix = np.cov(X.T)  
        self.eig_values, self.eig_vectors = ln.eig(cov_matrix)
        self.is_fit = True
    
    def get_params(self):
        if not self.is_fit:
            raise ValueError("PCA is not fit!")
        params_dict = {
            'eig_values': self.eig_values, 
            'eig_vectors': np.matrix(self.eig_vectors)
        }
        return params_dict
    
    def get_pca_component(self, percent):
        if not self.is_fit:
            raise ValueError("PCA is not fit!")
        for i in range(len(self.eig_values)):
            if sum(self.eig_values[0:i+1])/sum(self.eig_values) > percent:
                return i + 1 

    def perform_t_squared(self, X, V):
        if not self.is_fit:
            raise ValueError("PCA is not fit!")
        X = np.matrix(X)
        V = np.matrix(V)
        components = V.shape[1]
        eiv_diag_matrix=np.matrix(ln.inv(np.diag(self.eig_values[0:components])))
        value = X*V*eiv_diag_matrix*V.T*X.T
        return value.item()
 
    def perform_reconstruction_error(self, X, V):
        if not self.is_fit:
            raise ValueError("PCA is not fit!")
        X = np.matrix(X)
        V = np.matrix(V)
        return ln.norm((X - X*V*V.T))
