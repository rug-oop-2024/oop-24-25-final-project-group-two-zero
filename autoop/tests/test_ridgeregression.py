import unittest
import numpy as np
from .ridgeregression import RidgeRegression

class TestRidgeRegression(unittest.TestCase):

    def setUp(self) -> None:
        self.model = RidgeRegression(alpha=1.0, fit_intercept=True, solver='auto')
        self.observations = np.array([[1, 2], [3, 4], [5, 6]])
        self.ground_truth = np.array([1, 2, 3])

    def test_fit(self):
        self.model.fit(self.observations, self.ground_truth)
        self.assertIn('coef_', self.model.parameters)
        self.assertIn('intercept_', self.model.parameters)
        self.assertEqual(self.model.parameters['coef_'].shape, (self.observations.shape[1],))
        self.assertIsInstance(self.model.parameters['intercept_'], float)

    def test_predict(self):
        self.model.fit(self.observations, self.ground_truth)
        predictions = self.model.predict(self.observations)
        self.assertEqual(predictions.shape, self.ground_truth.shape)
        self.assertTrue(np.allclose(predictions, self.ground_truth, atol=1e-1))

if __name__ == '__main__':
    unittest.main()