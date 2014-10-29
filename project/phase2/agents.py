'''
The agent base class as well as the baseline agents.
'''

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
        return value*prob > price
    
    def fit_a_classifier(self, X_train, y_train, X_validation, y_validation):
        abstract()
    
    def predict_prob_of_good(self, x):
        return self.classifier.predict_proba(x)[0][1]
    

class RatioAgent(Agent):
    """Buys if the ratio of the price to value is below a specified threshold"""
    
    def __init__(self, name, max_p_v_ratio):
        super(RatioAgent, self).__init__(name)
        self.max_p_v_ratio = max_p_v_ratio
    
    def fit_a_classifier(self, X_train, y_train, X_validation, y_validation):
        pass
    
    def predict_prob_of_good(self, x):
        return 0
    
    def will_buy(self, value, price, prob):
        return (float(price)/value <= self.max_p_v_ratio)

from sklearn.naive_bayes import BernoulliNB

class NaiveBayesAgent(Agent):
    
    def fit_a_classifier(self, X_train, y_train, X_validation, y_validation):
        self.classifier = BernoulliNB()
        self.classifier.fit(X_train, y_train)

from sklearn.linear_model import LogisticRegression

class LRAgent(Agent):
    
    def fit_a_classifier(self, X_train, y_train, X_validation, y_validation):
        self.classifier = LogisticRegression()
        self.classifier.fit(X_train, y_train)

from sklearn.svm import SVC

class RBFAgent(Agent):
    
    def fit_a_classifier(self, X_train, y_train, X_validation, y_validation):
        self.classifier = SVC(C=1, probability=True)
        self.classifier.fit(X_train, y_train)
        
class PolyAgent(Agent):
    
    def fit_a_classifier(self, X_train, y_train, X_validation, y_validation):
        self.classifier = SVC(C=1, probability=True, kernel='poly')
        self.classifier.fit(X_train, y_train)


