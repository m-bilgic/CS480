'''
Created on Oct 20, 2014

@author: mbilgic
'''
from __future__ import print_function

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import cross_validation



from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

import numpy as np

def generate_binary_data(n_samples = 1000, l0_prob = 0.5, n_useful = 20, usefullness = (1000,1000), n_replicated = 0, n_random=0, seed=0):
    rg = np.random.RandomState(seed)
    
    n_features = n_useful + n_replicated + n_random
    
    u_f_p = [[],[]]  
    
    for _ in range(n_useful):
        u_f_p[0].append(rg.beta(*usefullness))
        u_f_p[1].append(rg.beta(*usefullness))
    
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=np.int)
    
    for i in range(n_samples):
        #sample a label
        label = int(rg.rand()<l0_prob)
        y[i] = label
        
        for f in range(n_useful):
            f_prob = u_f_p[label][f]
            X[i,f] = int(rg.rand() < f_prob)             
      
    return X, y

if __name__ == '__main__':
    
    n = 2000
    
    X, y = generate_binary_data(n_samples = n, n_useful = 100, usefullness=(50,50))  
    #X, y = make_classification(n_samples = n)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 0],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    
    scores = ['accuracy']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
    
        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score)
        clf.fit(X_train, y_train)
    
        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))
        print()
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()