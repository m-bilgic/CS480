'''
Created on Nov 11, 2014

@author: mbilgic
'''

import numpy as np
from sklearn.datasets import dump_svmlight_file

if __name__ == '__main__':
    
    data_path = './../phase2/'
    data_group = "dataset11"
    seed = 0
    save = True
    
    X_tr = np.loadtxt(data_path + data_group +  "_X_train.csv", dtype=float, delimiter=',')
    X_val = np.loadtxt(data_path + data_group +  "_X_val.csv", dtype=float, delimiter=',')
    y_tr = np.loadtxt(data_path + data_group +  "_y_train.csv", dtype=float, delimiter=',')
    y_val = np.loadtxt(data_path + data_group +  "_y_val.csv", dtype=float, delimiter=',')
    
    X = np.vstack((X_tr, X_val))
    y = np.hstack((y_tr, y_val))
    
    print X.shape
    print y.shape
    print np.sum(y)
    
    #exit(0)
    
    rs = np.random.RandomState(seed)
    
    rnd_indices = rs.permutation(X.shape[0])
    
    X = X[rnd_indices]
    y = y[rnd_indices]
    
    value = 1000.
    
    n = X.shape[0]
    
    prices = np.zeros(n)
    
    for i in range(n):
        prices[i] = rs.rand()*value
 
    if save:
        print "Saving dataset: %s" %data_group      
        
        np.savetxt(data_group+"_X.csv", X, delimiter=',', fmt='%0.5f')
        
        np.savetxt(data_group+"_y.csv", y, delimiter=',', fmt='%d')
        
        np.savetxt(data_group+"_p.csv", prices, delimiter=',', fmt='%0.2f')
        
        dump_svmlight_file(X, y, data_group+"_X_y.txt")
    
    