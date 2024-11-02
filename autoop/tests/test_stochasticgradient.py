import unittest
import numpy as np
from .stoasticgradient import StochasticGradient

class TestStochasticGradient(unittest.TestCase):

    def setUp(self) -> None:
        self.model = StochasticGradient()
        self.observations = np.array([[1, 2], [3, 4], [5, 6]])
        self.ground_truth = np.array([0, 1, 0])

    def test_fit(self):
        self.model.fit(self.observations, self.ground_truth)
        self.assertIn('coef_', self.model.parameters)
        self.assertIn('intercept_', self.model.parameters)
        self.assertEqual(self.model._model.coef_.shape[1], self.observations.shape[1])

    def test_predict(self):
        self.model.fit(self.observations, self.ground_truth)
        predictions = self.model.predict(self.observations)
        self.assertEqual(predictions.shape[0], self.observations.shape[0])
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))

if __name__ == '__main__':
    unittest.main()