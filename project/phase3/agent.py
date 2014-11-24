'''
The agent base class.
'''

def abstract():
    import inspect
    caller = inspect.getouterframes(inspect.currentframe())[1][3]
    raise NotImplementedError(caller + ' must be implemented in subclass')


class Agent(object):
    def __init__(self, name):
        """The constructor."""
        self.name = name
    
    def __repr__(self):
        return "Agent_" + self.name
    
    def train(self, X_train, y_train):
        """Train a naive Bayes classifier, using Laplace smoothing. Do not use scikitlearn.
        Assume that each feature is binary [0,1] and the class is binary [0,1].
        Do not hard code the number of features."""
        #abstract()
        pass
    
    def predict_prob_of_good(self, x):
        "Predict and return the probability of being Good (i.e., label 1)."
        #abstract()
        return 0.5