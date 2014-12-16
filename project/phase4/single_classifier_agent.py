from agents import Agent

# from sklearn.naive_bayes import BernoulliNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier


import numpy as np

def get_class( kls ):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)            
    return m

class ClassifierAgent(Agent):
    
    def __init__(self, name, clf_s="sklearn.naive_bayes.BernoulliNB"):
        self.name = name
        self.my_products = []
        self.product_labels = []
        self.rs = np.random.RandomState(0)
        self.clf_c = get_class(clf_s)
    
    def __repr__(self):
        return "Agent_" + self.name
    
    def choose_one_product(self, products):
        candidates = self.rs.permutation(len(products))
        max_exp_profit = float("-inf")
        chosen = -1
        for i in candidates[:100]:
            product = products[i]
            prob_good = self.clf.predict_proba(product.features)[0][1]

            exp_profit = prob_good*product.value - product.price
            if max_exp_profit < exp_profit:
                max_exp_profit = exp_profit 
                chosen = i
        
        if chosen != -1:
            return chosen
        else:
            return self.rs.choice(range(len(products)))
        
    def add_to_my_products(self, product, label):
        self.my_products.append(product)
        self.product_labels.append(label)
        if not hasattr(self, 'X'):
            self.X = product.features
            self.y = np.array([label], dtype=int)
        else:
            self.X = np.vstack((self.X, product.features))
            self.y = np.append(self.y, label)
        
        if len(self.y) > 1:
            self.clf = self.clf_c() # call the default constructor            
            self.clf.fit(self.X, self.y)