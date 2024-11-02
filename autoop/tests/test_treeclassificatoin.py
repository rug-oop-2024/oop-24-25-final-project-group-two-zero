import unittest
import numpy as np
from .treeclassification import TreeClassification

class TestTreeClassification(unittest.TestCase):

    def setUp(self) -> None:
        self.model = TreeClassification(criterion='gini', max_depth=3)
        self.observations = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.ground_truth = np.array([0, 1, 0, 1])

    def test_fit(self):
        self.model.fit(self.observations, self.ground_truth)
        self.assertIn('feature_importances_', self.model.parameters)
        self.assertEqual(len(self.model.parameters['feature_importances_']), self.observations.shape[1])

    def test_predict(self):
        self.model.fit(self.observations, self.ground_truth)
        predictions = self.model.predict(self.observations)
        self.assertEqual(predictions.shape[0], self.observations.shape[0])
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))

if __name__ == '__main__':
    unittest.main()