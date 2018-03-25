import unittest

from decision_stump import DecisionStump

class TestDecisionStump(unittest.TestCase):
    def test_train_and_predict(self):
        ds = DecisionStump()
        trainset = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
        labels = [1, -1, -1, 1]
        ds.train(trainset, labels)
        self.assertEqual(ds.dim, 0)
        self.assertEqual(ds.thresh, 0.0)
        self.assertEqual(ds.sign, -1)
        pred = ds.predict([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        self.assertEqual(pred, labels)

if __name__ == '__main__':
    unittest.main()