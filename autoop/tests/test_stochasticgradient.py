import numpy as np
from .stochasticgradient import StochasticGradient
from .base_classification_test_case import BaseClassificationTestCase


class TestStochasticGradient(BaseClassificationTestCase):
    model_class = StochasticGradient
    model_params = {}

    def check_fit_parameters(self):
        self.assertIn("coef_", self.model.parameters)
        self.assertIn("intercept_", self.model.parameters)
        self.assertEqual(self.model._model.coef_.shape[1], self.observations.shape[1])

    def check_predictions(self, predictions):
        self.assertEqual(predictions.shape[0], self.observations.shape[0])
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
