import numpy as np

from agent import Agent

if __name__ == '__main__':
    # Change these as needed
    data_path = "./"
    data_group = "dataset1"
       
    X_train_file = data_path + data_group +  "_X_train.csv"
    y_train_file = data_path + data_group + "_y_train.csv"   
    X_test_file = data_path + data_group + "_X_test.csv"
    y_test_file = data_path + data_group + "_y_test.csv"
    
    X_train = np.loadtxt(X_train_file, dtype=float, delimiter=',')
    y_train = np.loadtxt(y_train_file, dtype=int, delimiter=',')
    X_test = np.loadtxt(X_test_file, dtype=float, delimiter=',')
    y_test = np.loadtxt(y_test_file, dtype=int, delimiter=',')
    
    
    # Change this to your agent
    agent = None
    
    # Train the agent    
    agent.train(X_train, y_train)
    
    # Test the agent on the train itself, just for debugging purposes
    
    num_correct = 0
    for i in range(len(y_train)):
        prob = agent.predict_prob_of_good(X_train[i])
        if prob > 0.5:
            if y_train[i] == 1:
                num_correct += 1
        else:
            if y_train[i] == 0:
                num_correct += 1
    
    print
    print "Agent %s predicted %d cases correctly on the training set" %(agent, num_correct)
    print
    
    # Test the agent on the train itself, just for debugging purposes
    
    num_correct = 0
    for i in range(len(y_test)):
        prob = agent.predict_prob_of_good(X_test[i])
        if prob > 0.5:
            if y_test[i] == 1:
                num_correct += 1
        else:
            if y_test[i] == 0:
                num_correct += 1
    
    print
    print "Agent %s predicted %d cases correctly on the test set" %(agent, num_correct)
    print
    
    