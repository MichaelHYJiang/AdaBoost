
import numpy as np

class DecisionStump(object):
    def __init__(self):
        self.dim = None
        self.thresh = None
    
    def train(self, trainset, label):
        
        # assert trainset is a list of lists of float numbers
        trainset = np.array(trainset)
        num_samples, num_dim = trainset.shape
        min_error = num_samples
        for i in range(num_dim):
            possibilities = np.unique(trainset[:, i])
            possibilities.sort()
            for j in range(len(possibilities) - 1):
                threshold = (possibilities[j] + possibilities[j + 1]) / 2
                error = 0
                for k in range(num_samples):
                    error += 0