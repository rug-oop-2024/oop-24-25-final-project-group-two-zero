from .. import Model
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class TreeClassification(Model):
    """
    Decision Tree Classifier
    """
    type = "classification"

    def __init__(self, criterion='gini', max_depth=None, **kwargs):
        '''
        Initialize the Decision Tree Classifier with hyperparameters.

        Args:
            criterion (str): The function to measure the quality of a split (`'gini'`, `'entropy'`).
            max_depth (int): The maximum depth of the tree.
        '''
        super().__init__(**kwargs)
        self.criterion = criterion
        self.max_depth = max_depth
        self._model = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        '''
        Fit the Decision Tree model.

        Args:
            observations (np.ndarray): Training data features.
            ground_truth (np.ndarray): Training data labels.
        '''
        observations = np.asarray(observations)
        ground_truth = np.asarray(ground_truth)
        self._model.fit(observations, ground_truth)
        self._parameters['feature_importances_'] = self._model.feature_importances_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        '''
        Predict using the Decision Tree model.

        Args:
            observations (np.ndarray): Observations to predict.

        Returns:
            np.ndarray: Predicted labels.
        '''
        observations = np.asarray(observations)
        return self._model.predict(observations)
