'''
Created on Oct 15, 2014
'''

import numpy as np

def abstract():
    import inspect
    caller = inspect.getouterframes(inspect.currentframe())[1][3]
    raise NotImplementedError(caller + ' must be implemented in subclass')


class Agent(object):
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return "Agent_" + self.name
    
    def will_buy(self, value, price, prob):
        abstract()

class HalfProbAgent(Agent):
    
    def will_buy(self, value, price, prob):
        return (prob > 0.5)

class RatioAgent(Agent):
    
    def __init__(self, name, min_v_p_ratio):
        super(RatioAgent, self).__init__(name)
        self.min_v_p_ratio = min_v_p_ratio
    
    def will_buy(self, value, price, prob):
        return (value/price >= self.min_v_p_ratio)

class RationalAgent(Agent):
    
    def will_buy(self, value, price, prob):
        return (value*prob > price)   

if __name__ == '__main__':     
    
    value = 1000.
    
    num_products = 1000
    
    seed = None
    
    rg = np.random.RandomState(seed)
    
    agents = []
    
    agents.append(HalfProbAgent("hp"))
    
    agents.append(RatioAgent("ratio_1", 1))
    agents.append(RatioAgent("ratio_2", 2))
    agents.append(RatioAgent("ratio_4", 4))  
    
    agents.append(RationalAgent("rational"))
        
    
    
    agent_wealths = {}
    
    for agent in agents:
        agent_wealths[agent] = 0
    
    for _ in range(num_products):
        # price is always lower than or equal to the value
        price = rg.rand()*value
        
        # prob of working
        prob = rg.rand()
        
        # working or not?
        working = prob > rg.rand()
        
        for agent in agents:
            if agent.will_buy(value, price, prob):
                agent_wealths[agent] -= price
                if working:
                    agent_wealths[agent] += value
    
    for agent in agents:
        #print "%s:\t$%0.2f" %(agent, agent_wealths[agent])
        print "{}:\t${:11,.2f}".format(agent, agent_wealths[agent])