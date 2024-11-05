import numpy as np
from .treeclassification import TreeClassification
from autoop.tests.base_classification_test_case import BaseClassificationTestCase

class TestTreeClassification(BaseClassificationTestCase):
    model_class = TreeClassification
    model_params = {'criterion': 'gini', 'max_depth': 3}

    def check_fit_parameters(self):
        self.assertIn('feature_importances_', self.model.parameters)
        self.assertEqual(
            len(self.model.parameters['feature_importances_']),
            self.observations.shape[1]
        )

    def check_predictions(self, predictions):
        self.assertEqual(predictions.shape[0], self.observations.shape[0])
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
