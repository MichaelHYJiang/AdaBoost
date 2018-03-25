import numpy as np

class DecisionStump(object):
    def __init__(self):
        self.dim = None
        self.thresh = None
        self.sign = None
    
    def train(self, trainset, label, weights=None):
        # assert trainset is a list of lists of float numbers
        # assert labels are +1 and -1
        trainset = np.array(trainset)
        num_samples, num_dim = trainset.shape
        if weights is None:
            weights = [1] * num_samples
        else:
            assert(len(weights) == num_samples)
        min_error = num_samples
        best_dim = 0
        best_thresh = 0
        best_sign = 1
        for i in range(num_dim):
            possibilities = np.unique(trainset[:, i])
            possibilities.sort()
            # print('possibilities:', possibilities)
            for j in range(len(possibilities) - 1):
                threshold = (possibilities[j] + possibilities[j + 1]) / 2
                # print('threshold:', threshold)
                for sign in [+1, -1]:
                    error = 0
                    # print('=')
                    for k in range(num_samples):
                        # print(int(label[k] != sign * int(int(trainset[k, i] <= threshold) * 2 - 1)))
                        error += int(label[k] != sign * int((int(trainset[k, i] <= threshold) - 0.5) * 2)) * weights[k]
                    if error < min_error:
                        min_error = error
                        best_dim = i
                        best_thresh = threshold
                        best_sign = sign
        # print('min_error:', min_error)
        self.dim = best_dim
        self.thresh = best_thresh
        self.sign = best_sign
        
    def predict(self, testset):
        # assert testset is a list of lists of float numbers
        # output labels are +1 and -1 in a list
        pred = []
        for sample in testset:
            prediction = (int(sample[self.dim] <= self.thresh) * 2 - 1) * self.sign
            pred.append(prediction)
        return pred
