# knn.py
from sklearn.neighbors import KNeighborsClassifier
from .model import Model
import numpy as np


class KNearestNeighbors(Model):
    """
    K-Nearest Neighbors classifier.
    """

    _type = "classification"
    _available_hyperparameters = {
        'n_neighbors': 5,
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': 30,
        'p': 2,  # Power parameter for the Minkowski metric
        'metric': ['minkowski', 'euclidean', 'manhattan'],
        'metric_params': None,
        'n_jobs': None
    }

    def __init__(self, **hyperparameters) -> None:
        """
        Initializes the KNearestNeighbors model with hyperparameters.

        Args:
            **hyperparameters: Hyperparameters for the model.
        """

        super().__init__(**hyperparameters)
        # Merge default hyperparameters with user-specified ones
        params = {k: self._hyperparameters.get(k, v) for k, v in self._available_hyperparameters.items()}
        self._model = KNeighborsClassifier(**params)

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
        Predict using the KNearestNeighbors model.

        Args:
            observations (np.ndarray): Observations to predict.

        Returns:
            np.ndarray: Predicted labels.
        """

        return self._model.predict(observations)
