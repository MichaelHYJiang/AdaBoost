import unittest

from decision_stump import DecisionStump

from Adaboost import Adaboost

class TestDecisionStump(unittest.TestCase):
    def test_train_and_predict(self):
        print('test DecisionStump')
        ds = DecisionStump()
        trainset = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
        labels = [1, -1, -1, 1]
        ds.train(trainset, labels)
        self.assertEqual(ds.dim, 0)
        self.assertEqual(ds.thresh, 0.0)
        self.assertEqual(ds.sign, -1)
        pred = ds.predict([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        self.assertEqual(pred, labels)
    
    def test_weight(self):
        ds = DecisionStump()
        trainset = [[1, 1], [1, -1], [1, -2], [-1, 1]]
        labels = [1, 1, -1, -1]
        weights = [2, 1, 3, 1]
        # print('test weights')
        ds.train(trainset, labels,weights)
        # print(ds.dim)
        # print(ds.thresh)
        # print(ds.sign)
        self.assertEqual(ds.dim, 1)
        self.assertEqual(ds.thresh, -1.5)
        
        
class TestAdaboost(unittest.TestCase):
    def test_train(self):
        print('test Adaboost')
        trainset = [[0],
                    [1],
                    [2],
                    [3],
                    [4],
                    [5],
                    [6],
                    [7],
                    [8],
                    [9]]
        labels = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
        ab = Adaboost(DecisionStump, 3)
        ab.train(trainset, labels)
        self.assertEqual(ab.beta, [0.42364893019360172, 0.64964149206513044, 0.75203869838813697])
        pred = ab.predict(trainset)
        self.assertEqual(list(pred), labels)

if __name__ == '__main__':
    unittest.main()