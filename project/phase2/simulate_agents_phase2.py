import sys
import numpy as np

from agents import RatioAgent, NaiveBayesAgent, LRAgent, RBFAgent, PolyAgent

def simulate_agents(agents, value, X, y, seed=None):
    
    agent_wealths = {}
    
    for agent in agents:
        agent_wealths[agent] = 0
    
    if seed:
        np.random.seed(seed)
    
    num_products = X_val.shape[0]
    
    for p in range(num_products):
        # price is always lower than or equal to the value
        price = np.random.rand()*value
         
        # working or not?
        working = (y[p] == 1)
        
        for agent in agents:
            if agent.will_buy(value, price, agent.predict_prob_of_good(X[p])):
                agent_wealths[agent] -= price
                if working:
                    agent_wealths[agent] += value
    
    return agent_wealths

if __name__ == '__main__': 
    #print len(sys.argv)
    #print sys.argv[0]
    data_path = "./"
    data_group = "dataset5"
    
    X_train_file = data_path + data_group +  "_X_train.csv"
    y_train_file = data_path + data_group + "_y_train.csv"   
    X_val_file = data_path + data_group + "_X_val.csv"
    y_val_file = data_path + data_group + "_y_val.csv"
    X_test_file = data_path + data_group + "_X_test.csv"
    y_test_file = data_path + data_group + "_y_test.csv"
    
    X_train = np.loadtxt(X_train_file, dtype=float, delimiter=',')
    y_train = np.loadtxt(y_train_file, dtype=int, delimiter=',')
    X_val = np.loadtxt(X_val_file, dtype=float, delimiter=',')
    y_val = np.loadtxt(y_val_file, dtype=int, delimiter=',')
    X_test = np.loadtxt(X_test_file, dtype=float, delimiter=',')
    y_test = np.loadtxt(y_test_file, dtype=int, delimiter=',')
    
    agents = []
    
    
    agents.append(RatioAgent("ratio_0.75", 0.75))
    agents.append(RatioAgent("ratio_0.50", 0.5))
    agents.append(RatioAgent("ratio_0.25", 0.25))
    agents.append(NaiveBayesAgent("nb"))
    agents.append(LRAgent("lr"))
    agents.append(RBFAgent("rbf"))
    agents.append(PolyAgent("poly"))
    
    # Train the agents
    for agent in agents:
        agent.fit_a_classifier(X_train, y_train, X_val, y_val)
    
    value = 1000
    agent_wealths = simulate_agents(agents, value, X_val, y_val)
    
    for agent in agents:
        print "{}:\t\t${:,.2f}".format(agent, agent_wealths[agent])
    
    
    