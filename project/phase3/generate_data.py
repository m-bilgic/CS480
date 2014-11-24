
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score


import numpy as np

def generate_binary_data(n_samples = 1000, l0_prob = 0.5, n_useful = 20, usefullness = (1000,1000), n_product = 0, n_replicate = 0, n_random=0, seed=0):
    rg = np.random.RandomState(seed)
    
    n_features = n_useful + n_replicate + n_product + n_random
    
    u_f_p = [[],[]]  
    
    for _ in range(n_useful):
        u_f_p[0].append(rg.beta(*usefullness))
        u_f_p[1].append(rg.beta(*usefullness))
    
    print u_f_p[0]
    print u_f_p[1]
    
    #exit(0)
    
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

if __name__ == '__main__':
    
    n = 2000
    l0_prob = 0.75
    seed = 3
    
    
    X, y = generate_binary_data(n_samples = n, l0_prob = l0_prob, n_useful = 10, n_product = 0, n_replicate = 0, n_random = 0, usefullness=(1,1), seed=seed)  
    
    save = True
    dataset = 4
        
    classifiers = []
    
    classifiers.append(BernoulliNB())
   
    
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
    
 
    
    if save:
        print "Saving dataset: %d" %dataset
        
        dataset = str(dataset)
            
        np.savetxt("dataset"+dataset+"_X_train.csv", X[:ts], delimiter=',', fmt='%0.5f')
        np.savetxt("dataset"+dataset+"_X_test.csv", X[ts:], delimiter=',', fmt='%0.5f')
        
        np.savetxt("dataset"+dataset+"_y_train.csv", y[:ts], delimiter=',', fmt='%d')
        np.savetxt("dataset"+dataset+"_y_test.csv", y[ts:], delimiter=',', fmt='%d')
    
    
    