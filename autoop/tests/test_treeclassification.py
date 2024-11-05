# test_tree_classification.py
import numpy as np
from .treeclassification import TreeClassification
from .base_test_case import BaseModelTestCase

class TestTreeClassification(BaseModelTestCase):
    model_class = TreeClassification
    model_params = {'criterion': 'gini', 'max_depth': 3}
    observations = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    ground_truth = np.array([0, 1, 0, 1])

    def check_fit_parameters(self):
        self.assertIn('feature_importances_', self.model.parameters)
        self.assertEqual(len(self.model.parameters['feature_importances_']), self.observations.shape[1])

    def check_predictions(self, predictions):
        self.assertEqual(predictions.shape[0], self.observations.shape[0])
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
