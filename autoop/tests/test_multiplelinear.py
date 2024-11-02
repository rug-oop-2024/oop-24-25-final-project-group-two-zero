import unittest
import numpy as np
from .multiplelinear import MultipleLinearRegression

class TestMultipleLinearRegression(unittest.TestCase):

    def setUp(self) -> None:
        self.model = MultipleLinearRegression()
        self.observations = np.array([[1, 2], [3, 4], [5, 6]])
        self.ground_truth = np.array([1, 2, 3])

    def test_fit(self):
        self.model.fit(self.observations, self.ground_truth)
        self.assertIn('weights', self.model.parameters)
        expected_weights = np.array([0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(self.model.parameters['weights'], expected_weights, decimal=1)

    def test_predict(self):
        self.model.fit(self.observations, self.ground_truth)
        predictions = self.model.predict(self.observations)
        expected_predictions = np.array([1, 2, 3])
        np.testing.assert_array_almost_equal(predictions, expected_predictions, decimal=1)

    def test_fit_without_intercept(self):
        model_no_intercept = MultipleLinearRegression(fit_intercept=False)
        model_no_intercept.fit(self.observations, self.ground_truth)
        self.assertIn('weights', model_no_intercept.parameters)
        expected_weights = np.array([0.2, 0.4])
        np.testing.assert_array_almost_equal(model_no_intercept.parameters['weights'], expected_weights, decimal=1)

    def test_predict_without_intercept(self):
        model_no_intercept = MultipleLinearRegression(fit_intercept=False)
        model_no_intercept.fit(self.observations, self.ground_truth)
        predictions = model_no_intercept.predict(self.observations)
        expected_predictions = np.array([1, 2, 3])
        np.testing.assert_array_almost_equal(predictions, expected_predictions, decimal=1)

    def test_fit_raises_value_error_on_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            self.model.fit(self.observations, np.array([1, 2]))

    def test_predict_raises_value_error_if_not_fitted(self):
        with self.assertRaises(ValueError):
            self.model.predict(self.observations)

if __name__ == '__main__':
    unittest.main()