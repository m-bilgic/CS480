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

def generate_binary_data(n_samples = 1000, l0_prob = 0.5, n_useful = 20, usefullness = (1000,1000), n_product = 0, n_replicate = 0, n_random=0, seed=0):
    rg = np.random.RandomState(seed)
    
    n_features = n_useful + n_replicate + n_product + n_random
    
    u_f_p = [[],[]]  
    
    for _ in range(n_useful):
        u_f_p[0].append(rg.beta(*usefullness))
        u_f_p[1].append(rg.beta(*usefullness))
    
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=np.int)    
    
    # useful features
    for i in range(n_samples):
        #sample a label
        label = int(rg.rand()>l0_prob)
        y[i] = label
        
        for f in range(n_useful):
            f_prob = u_f_p[label][f]
            X[i,f] = int(rg.rand() < f_prob)
    
    f_index = n_useful
    
    # replicate the useful features
    for f in range(f_index, f_index + n_replicate):
        r = rg.randint(n_useful)
        for i in range(n_samples):
            X[i,f] = X[i,r]
    
    f_index += n_replicate
    
    # take the product of two features
    for f in range(f_index, f_index + n_product):
        f1 = rg.randint(f_index)
        f2 = rg.randint(f_index)
        for i in range(n_samples):
            X[i,f] = X[i,f1]*X[i,f2]
    
    f_index += n_product
    
    # random features
    for f in range(f_index, f_index + n_random):
        for i in range(n_samples):
            X[i,f] = rg.randint(2)   
      
    return X, y

def generate_labels(X, classifier, l0_prob=0.5):
    probs = classifier.predict_proba(X)
    indices = np.argsort(probs[:,1])
    n_samples = X.shape[0] 
    num_l0 = int(n_samples*l0_prob)
    y = np.zeros(n_samples, dtype=np.int)
    y[indices[:num_l0]] = 0
    y[indices[num_l0:]] = 1
    return y

if __name__ == '__main__':
    
    n = 3000
    l0_prob = 0.75
    
    
    #X, y = generate_binary_data(n_samples = n, l0_prob = l0_prob, n_useful = 40, n_product = 20, n_replicate = 20, n_random = 20, usefullness=(50,50))  
    seed = 12
    preferred_clf = 3
    save = True
    dataset = 12
    
    X, y = make_classification(class_sep=.5, n_samples = n, n_features = 20, n_informative = 10, n_redundant = 2, n_repeated = 2, n_clusters_per_class = 3, weights=[l0_prob, 1-l0_prob], random_state=seed)
    
    classifiers = []
    
    classifiers.append(BernoulliNB())
    classifiers.append(LogisticRegression(C=1))
    classifiers.append(svm.SVC(C=1, probability=True))
    classifiers.append(svm.SVC(C=1, probability=True, kernel='poly'))
    
    
    ts = 1000
    
    X_train = X[:ts]
    y_train = y[:ts]
    
    X_test = X[ts:]
    y_test = y[ts:]
    
    
    max_accu = 0
    the_classifier = None
    
    for clf in classifiers:
        clf.fit(X_train, y_train)
        y_true, y_pred = y_test, clf.predict(X_test)
       
        accu = accuracy_score(y_true, y_pred)
        if max_accu < accu:
            max_accu = accu
            the_classifier = clf 
        print clf
        print accu
        print
    
    
    print
    print
    print "The classifier"
    print the_classifier
    print max_accu
    
    print
    print
    print "Modifying the labels"
    
    # Modify only the test labels
    y_test = generate_labels(X[ts:], classifiers[preferred_clf], l0_prob=l0_prob)
    
    from agents import RatioAgent, NaiveBayesAgent, LRAgent, RBFAgent, PolyAgent
    from simulate_agents_phase2 import simulate_agents
    
    agents = []
    
    
    agents.append(RatioAgent("ratio_0.75", 0.75))
    agents.append(RatioAgent("ratio_0.50", 0.5))
    agents.append(RatioAgent("ratio_0.25", 0.25))
    agents.append(NaiveBayesAgent("nb"))
    agents.append(LRAgent("lr"))
    agents.append(RBFAgent("rbf"))
    agents.append(PolyAgent("poly"))
    
    X_val = X[ts:2*ts]
    y_val = y_test[:ts]
    
    # Train the agents
    for agent in agents:
        agent.fit_a_classifier(X_train, y_train, X_val, y_val)
    
    value = 1000
    agent_wealths = simulate_agents(agents, value, X_val, y_val)
    
    for agent in agents:
        print "{}:\t\t${:,.2f}".format(agent, agent_wealths[agent])
    
    #print np.sum(y_test)
    
    # Modify both the train and test labels
    #y_new = generate_labels(X, classifiers[3], l0_prob=l0_prob)
    #print np.sum(y != y_new)
    #exit(0)
    #y_train = y_new[:ts]
    #y_test = y_new[ts:]
    
    #print np.sum(y_train), np.sum(y_test)
    #exit(0)
    
    #classifiers = []
    
    #classifiers.append(BernoulliNB())
    #classifiers.append(LogisticRegression(C=1))
    #classifiers.append(svm.SVC(C=1, probability=True))
    #classifiers.append(svm.SVC(C=1, probability=True, kernel='poly'))
    
    
    min_error = X_test.shape[0]
    
    max_accu = 0
    the_classifier = None
    
    for clf in classifiers:
        #clf.fit(X_train, y_train)
        
        probs = clf.predict_proba(X_test)
        
        error = 0
        for i in range(X_test.shape[0]):
            error += (1 - probs[i][y_test[i]])
        
        if min_error > error:
            min_error = error
            the_classifier = clf 
        print clf
        print error
        print
    
    
    print
    print
    print "The final classifier"
    print the_classifier
    print min_error    
    
    if save:
        print "Saving dataset: %d" %dataset
        
        dataset = str(dataset)
            
        np.savetxt("dataset"+dataset+"_X_train.csv", X[:ts], delimiter=',', fmt='%0.5f')
        np.savetxt("dataset"+dataset+"_X_val.csv", X[ts:2*ts], delimiter=',', fmt='%0.5f')
        np.savetxt("dataset"+dataset+"_X_test.csv", X[2*ts:], delimiter=',', fmt='%0.5f')
        
        np.savetxt("dataset"+dataset+"_y_train.csv", y[:ts], delimiter=',', fmt='%d')
        np.savetxt("dataset"+dataset+"_y_val.csv", y_test[:ts], delimiter=',', fmt='%d')
        np.savetxt("dataset"+dataset+"_y_test.csv", y_test[ts:], delimiter=',', fmt='%d')
    
    
    