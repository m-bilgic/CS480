'''
Created on Nov 19, 2014
'''

import numpy as np

import matplotlib.pyplot as plt


def prob1(x, w0, w):
    'Return the probability of class 1'
    esum = np.exp(w0 + np.sum(x*w))
    return esum / (1+esum)

def grad0(X, y, w0, w):
    'Compute the gradient for w0'
    gsum = 0
    num_data = len(y)
    for i in range(num_data):
        if y[i] == 0:
            gsum -= (prob1(X[i], w0, w))
        else:
            gsum += (1-prob1(X[i], w0, w))
    return gsum


def gradj(X, y, w0, w, j):
    'Compute the gradient for jth feature'
    gsum = 0
    num_data = len(y)
    for i in range(num_data):
        if y[i] == 0:
            gsum -= X[i][j]*(prob1(X[i], w0, w))
        else:
            gsum += X[i][j]*(1-prob1(X[i], w0, w))
    return gsum


def print_probs(X, y, w0, w):
    num_data = len(X)
    for i in range(num_data):
        print X[i], y[i], prob1(X[i], w0, w)

def cll(X, y, w0, w):
    cll_sum = 0.
    num_data = len(X)
    for i in range(num_data):
        if y[i] == 0:
            cll_sum += np.log(1-prob1(X[i], w0, w))
        else:
            cll_sum += np.log(prob1(X[i], w0, w))
    return cll_sum

def objective_l2(X, y, w0, w, C):
    penalty = w0*w0 + np.sum(w*w)
    return C*cll(X, y, w0, w) - penalty


if __name__ == '__main__':
    
    # Data        
    X = np.array([[-3], [-2], [-1], [1], [2], [3]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    # Initialize    
    w0=0.
    w=np.array([0.])
    
    # Learning rate
    lr=0.5
    
    # Complexity param
    C=1.
    
    # Print probs
    print "Probabilities after initialization."    
    print_probs(X, y, w0, w)
    
    
    print "\nPerforming gradient ascent - no regularization"
    
    num_iter = 10
    for it in range(num_iter):
        print "\nIteration: %d" %it
        w0 += lr*grad0(X, y, w0, w)
        print "w0: %f" %w0
        for j in range(len(w)):
            w[j] += lr*gradj(X, y, w0, w, j)
            print "w%d: %f" %((j+1), w[j])
        print "Probs"
        print_probs(X, y, w0, w)
        print "CLL: %f" %cll(X, y, w0, w)
        print "Objective: %f" %objective_l2(X, y, w0, w, C)
    
    x = np.linspace(-6,6,num=100)
    plt.plot(x, [prob1(x_, w0, w) for x_ in x], 'r', linewidth=3)
    
    
    
    # Initialize    
    w0=0.
    w=np.array([0.])
    
    # Learning rate
    lr=0.5
    
    # Complexity param
    C=.5
    num_iter = 10
    print "\nPerforming gradient ascent - with l2 regularization"
    lr = min([0.5, 0.5/C])
    for it in range(num_iter):
        print "\nIteration: %d" %it
        w0 += C*lr*grad0(X, y, w0, w) - lr*w0
        print "w0: %f" %w0
        for j in range(len(w)):
            w[j] += C*lr*gradj(X, y, w0, w, j) - lr*w[j]
            print "w%d: %f" %((j+1), w[j])
        print "Probs"
        print_probs(X, y, w0, w)
        print "CLL: %f" %cll(X, y, w0, w)
        print "Objective: %f" %objective_l2(X, y, w0, w, C)
    
    plt.plot(x, [prob1(x_, w0, w) for x_ in x], 'g', linewidth=3)
    
    # Initialize    
    w0=0.
    w=np.array([0.])
    
    # Learning rate
    lr=0.5
    
    # Complexity param
    C=1
    num_iter = 10
    print "\nPerforming gradient ascent - with l2 regularization"
    lr = min([0.5, 0.5/C])
    for it in range(num_iter):
        print "\nIteration: %d" %it
        w0 += C*lr*grad0(X, y, w0, w) - lr*w0
        print "w0: %f" %w0
        for j in range(len(w)):
            w[j] += C*lr*gradj(X, y, w0, w, j) - lr*w[j]
            print "w%d: %f" %((j+1), w[j])
        print "Probs"
        print_probs(X, y, w0, w)
        print "CLL: %f" %cll(X, y, w0, w)
        print "Objective: %f" %objective_l2(X, y, w0, w, C)
    
    plt.plot(x, [prob1(x_, w0, w) for x_ in x], 'b', linewidth=3)
    
    
    # Initialize    
    w0=0.
    w=np.array([0.])
    
    # Learning rate
    lr=0.5
    
    # Complexity param
    C=10
    num_iter = 10
    print "\nPerforming gradient ascent - with l2 regularization"
    lr = min([0.5, 0.5/C])
    for it in range(num_iter):
        print "\nIteration: %d" %it
        w0 += C*lr*grad0(X, y, w0, w) - lr*w0
        print "w0: %f" %w0
        for j in range(len(w)):
            w[j] += C*lr*gradj(X, y, w0, w, j) - lr*w[j]
            print "w%d: %f" %((j+1), w[j])
        print "Probs"
        print_probs(X, y, w0, w)
        print "CLL: %f" %cll(X, y, w0, w)
        print "Objective: %f" %objective_l2(X, y, w0, w, C)
    
    plt.plot(x, [prob1(x_, w0, w) for x_ in x], 'k', linewidth=3)
    

    
    
    
    plt.show()