import numpy as np
from .ridgeregression import RidgeRegression
from .base_regression_test_case import BaseRegressionTestCase


class TestRidgeRegression(BaseRegressionTestCase):
    model_class = RidgeRegression
    model_params = {"alpha": 1.0, "fit_intercept": True, "solver": "auto"}

    def check_fit_parameters(self):
        self.assertIn("coef_", self.model.parameters)
        self.assertIn("intercept_", self.model.parameters)
        self.assertEqual(
            self.model.parameters["coef_"].shape, (self.observations.shape[1],)
        )
        self.assertIsInstance(self.model.parameters["intercept_"], float)

    def check_predictions(self, predictions):
        self.assertEqual(predictions.shape, self.ground_truth.shape)
        self.assertTrue(np.allclose(predictions, self.ground_truth, atol=1e-1))
