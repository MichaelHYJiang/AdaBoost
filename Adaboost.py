import numpy as np

from decision_stump import DecisionStump

class Adaboost:
    def __init__(self, weak_classifier, max_iter=10, thresh=1e-4):
        self.weak_classifier = weak_classifier
        self.classifiers = []
        self.beta = []
        self.max_iter = max_iter
        self.thresh = thresh
    
    def train(self, trainset, labels, max_iter=None):
        if max_iter is None:
            max_iter = self.max_iter
        weights = np.array([1] * len(trainset)) / len(trainset)
        labels = np.array(labels)
        for i in range(max_iter):
            classifier = self.weak_classifier()
            classifier.train(trainset, labels, weights)
            pred = np.array(classifier.predict(trainset))
            error_rate = np.inner(weights, np.array(pred != labels, dtype='int'))
            # print('er:', error_rate)
            beta = 0.5 * np.log(1 / error_rate - 1)
            self.beta.append(beta)
            self.classifiers.append(classifier)
            # print(np.exp(-beta * labels * pred) * weights)
            weights = weights * np.exp(-beta * labels * pred)
            weights = weights / weights.sum()
            # print('weights:', weights)
            
    def predict(self, testset):
        pred = []
        for sample in testset:
            prediction = np.inner(self.beta, [classifier.predict([sample])[0] for classifier in self.classifiers])
            pred.append(prediction)
        return (np.array(pred) > self.thresh).astype('int') * 2 - 1
        
def cross_validation(trainset, labels, classifier):
    pass
    
if __name__ == '__main__':
    trainset = [[0, 1, 3],
                [0, 3, 1],
                [1, 2, 2],
                [1, 1, 3],
                [1, 2, 3],
                [0, 1, 2],
                [1, 1, 2],
                [1, 1, 1],
                [1, 3, 1],
                [0, 2, 1]]
    labels = [-1, -1, -1, -1, -1, -1, 1, 1, -1, -1]
    ada = Adaboost(DecisionStump, max_iter=3)
    ada.train(trainset, labels)
    pred = ada.predict(trainset)
    print((pred != np.array(labels)).astype('int').sum() / len(labels))
    print(pred)
    print(np.array(labels))
    print(ada.beta)
