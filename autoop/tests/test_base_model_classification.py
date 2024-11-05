# base_classification_test_case.py
import unittest
import numpy as np

class BaseClassificationTestCase(unittest.TestCase):
    model_class = None
    model_params = {}
    observations = np.array([
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1],
        [2, 2],
        [2, -2],
        [-2, 2],
        [-2, -2]
    ])
    ground_truth = np.array([0, 0, 0, 1, 0, 0, 0, 1])

    def setUp(self):
        self.model = self.model_class(**self.model_params)

    def test_fit(self):
        self.model.fit(self.observations, self.ground_truth)
        self.check_fit_parameters()

    def test_predict(self):
        self.model.fit(self.observations, self.ground_truth)
        predictions = self.model.predict(self.observations)
        self.check_predictions(predictions)

    def check_fit_parameters(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def check_predictions(self, predictions):
        raise NotImplementedError("Subclasses should implement this method.")
