class Adaboost:
    def __init__(self, weak_classifier, max_iter=10):
        self.weak_classifier = weak_classifier
        self.classifiers = []
        self.beta = []
        self.max_iter = max_iter
    
    def train(self, trainset, labels, max_iter=self.max_iter):
        pass
