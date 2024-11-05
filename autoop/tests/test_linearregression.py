import numpy as np
from .LinearRegression import LinearRegressionModel
from .base_regression_test_case import BaseRegressionTestCase

class TestLinearRegressionModel(BaseRegressionTestCase):
    model_class = LinearRegressionModel
    model_params = {}
    test_observations = np.array([[6], [7]])
    expected_predictions = np.array([12, 14])

    def check_fit_parameters(self):
        self.assertIn('coef_', self.model.parameters)
        self.assertIn('intercept_', self.model.parameters)
        np.testing.assert_array_almost_equal(
            self.model.parameters['coef_'],
            np.array([2.0]),
            decimal=1
        )
        self.assertAlmostEqual(self.model.parameters['intercept_'], 0.0, places=1)

    def check_predictions(self, predictions):
        np.testing.assert_array_almost_equal(
            predictions,
            self.expected_predictions,
            decimal=1
        )
