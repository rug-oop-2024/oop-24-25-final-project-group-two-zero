import unittest
import numpy as np
from .LinearRegression import LinearRegressionModel

class TestLinearRegressionModel(unittest.TestCase):

    def setUp(self) -> None:
        self.model = LinearRegressionModel()
        self.observations = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        self.ground_truth = np.array([2, 4, 6, 8])

    def test_fit(self):
        self.model.fit(self.observations, self.ground_truth)
        self.assertIn('coef_', self.model.parameters)
        self.assertIn('intercept_', self.model.parameters)
        np.testing.assert_array_almost_equal(self.model.parameters['coef_'], np.array([2, 2]))
        self.assertAlmostEqual(self.model.parameters['intercept_'], 0)

    def test_predict(self):
        self.model.fit(self.observations, self.ground_truth)
        predictions = self.model.predict(np.array([[5, 5], [6, 6]]))
        np.testing.assert_array_almost_equal(predictions, np.array([10, 12]))

if __name__ == '__main__':
    unittest.main()