from .. import Model
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class TreeClassification(Model):
    """
    Decision tree classification model.
    """
    def __init__(self, criterion='gini', max_depth=None, **kwargs) -> None:
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

    def fit(self, observations: np.ndarray, groundtruth: np.ndarray) -> None:
        """
        Fits the model to the data.

        Args:
            observations (np.ndarray): Features.
            groundtruth (np.ndarray): Target values.
        """
        observations = np.asarray(observations)
        ground_truth = np.asarray(ground_truth)
        self._model.fit(observations, ground_truth)
        self._parameters['feature_importances_'] = self._model.feature_importances_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict using the Decision Tree Classifier model.

        Args:
            observations (np.ndarray): Observations to predict.

        Returns:
            np.ndarray: Predicted labels.
        """
        observations = np.asarray(observations)
        return self._model.predict(observations)