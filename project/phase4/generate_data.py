'''
Created on Oct 20, 2014

@author: mbilgic
'''
#from __future__ import print_function


from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

import numpy as np


if __name__ == '__main__':
    
    n = 3000
    l0_prob = 0.50
    
    
    #X, y = generate_binary_data(n_samples = n, l0_prob = l0_prob, n_useful = 40, n_product = 20, n_replicate = 20, n_random = 20, usefullness=(50,50))  
    seed = 1
    save = True
    dataset = 1
    
    X, y = make_classification(class_sep=.5, n_samples = n, n_features = 20, n_informative = 10, n_redundant = 2, n_repeated = 2, n_clusters_per_class = 3, weights=[l0_prob, 1-l0_prob], random_state=seed)
    
    rs = np.random.RandomState(seed)
    
    value = 100
    
    prices = np.zeros(n)
    
    for i in range(n):
        prices[i] = rs.rand()*value
 
    if save:
        print "Saving dataset: %d" %dataset
        
        dataset = str(dataset)
            
        np.savetxt("dataset"+dataset+"_X.csv", X, delimiter=',', fmt='%0.5f')
        
        np.savetxt("dataset"+dataset+"_y.csv", y, delimiter=',', fmt='%d')
        
        np.savetxt("dataset"+dataset+"_p.csv", prices, delimiter=',', fmt='%0.2f')
        
    
    
    