import numpy as np
from .multiplelinear import MultipleLinearRegression
from .base_regression_test_case import BaseRegressionTestCase


class TestMultipleLinearRegression(BaseRegressionTestCase):
    model_class = MultipleLinearRegression
    model_params = {}

    def check_fit_parameters(self):
        self.assertIn("weights", self.model.parameters)
        expected_weights = np.array([2.0, 0.0])  # Assuming model adds intercept
        np.testing.assert_array_almost_equal(
            self.model.parameters["weights"], expected_weights, decimal=1
        )

    def check_predictions(self, predictions):
        expected_predictions = self.ground_truth
        np.testing.assert_array_almost_equal(
            predictions, expected_predictions, decimal=1
        )

    # Additional tests specific to MultipleLinearRegression
    def test_fit_without_intercept(self):
        model_no_intercept = self.model_class(fit_intercept=False)
        model_no_intercept.fit(self.observations, self.ground_truth)
        self.assertIn("weights", model_no_intercept.parameters)
        expected_weights = np.array([2.0])  # Without intercept
        np.testing.assert_array_almost_equal(
            model_no_intercept.parameters["weights"], expected_weights, decimal=1
        )

    def test_predict_without_intercept(self):
        model_no_intercept = self.model_class(fit_intercept=False)
        model_no_intercept.fit(self.observations, self.ground_truth)
        predictions = model_no_intercept.predict(self.observations)
        expected_predictions = self.ground_truth
        np.testing.assert_array_almost_equal(
            predictions, expected_predictions, decimal=1
        )

    def test_fit_raises_value_error_on_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            self.model.fit(self.observations, np.array([1, 2]))

    def test_predict_raises_value_error_if_not_fitted(self):
        with self.assertRaises(ValueError):
            self.model.predict(self.observations)
