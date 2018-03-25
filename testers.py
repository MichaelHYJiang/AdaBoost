import unittest

from decision_stump import DecisionStump

class TestDecisionStump(unittest.TestCase):
    def test_train(self):
        ds = DecisionStump()
        trainset = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
        labels = [1, -1, -1, 1]
        ds.train(trainset, labels)
        self.assertEqual(ds.dim, 0)
        self.assertEqual(ds.thresh, 0.0)
        self.assertEqual(ds.sign, -1)

if __name__ == '__main__':
    unittest.main()