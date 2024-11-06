from sklearn.tree import DecisionTreeClassifier
from ..model import Model
import numpy as np


class TreeClassification(Model):
    """
    Decision Tree Classifier.
    """

    _type = "classification"
    _available_hyperparameters = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random'],
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': None,
    }
    _supported_feature_types = ['numerical']
    _supported_target_types = ['categorical']

    def __init__(self, **hyperparameters) -> None:
        """
        Initializes the TreeClassification model with hyperparameters.

        Args:
            **hyperparameters: Hyperparameters for configuring the DecisionTreeClassifier.
        """
        super().__init__(**hyperparameters)
        params = {k: self._hyperparameters.get(k, v) for k, v in self._available_hyperparameters.items()}
        self._model = DecisionTreeClassifier(**params)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the model to the data.

        Args:
            observations (np.ndarray): Features.
            ground_truth (np.ndarray): Target values.
        """
        self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict using the TreeClassification model.

        Args:
            observations (np.ndarray): Observations to predict.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self._model.predict(observations)
