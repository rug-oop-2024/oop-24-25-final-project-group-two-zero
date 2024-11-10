from sklearn.neighbors import KNeighborsClassifier
from ..model import Model
import numpy as np
from typing import Any


class KNearestNeighbors(Model):
    """
    K-Nearest Neighbors classifier.
    """
    _type: str = "classification"
    _available_hyperparameters: dict = {
        "n_neighbors": 5,
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": 30,
        "p": 2.0,
        "metric": ["minkowski", "euclidean", "manhattan"],
        "metric_params": None,
        "n_jobs": None,
    }
    supported_feature_types: list = ["numerical"]
    supported_target_types: list = ["cataorical"]

    def __init__(self, **hyperparameters: Any) -> None:
        """
        Initializes the KNearestNeighbors model
        with hyperparameters.

        Args:
            **hyperparameters: Hyperparameters for the model.
        """

        super().__init__(**hyperparameters)
        # Merge default hyperparameters with user-specified ones
        self._parameters = {
            k: self._hyperparameters.get(k, v)
            for k, v in self._available_hyperparameters.items()
        }
        self._model = KNeighborsClassifier(**self._parameters)

    def fit(
        self,
        observations: np.ndarray,
        ground_truth: np.ndarray
        ) -> None:
        """
        Fits the model to the data.

        Args:
            observations (np.ndarray): Features.
            ground_truth (np.ndarray): Target values.
        """
        ground_truth = ground_truth.ravel()
        self._model.fit(observations, ground_truth)
        self._parameters = {
            "n_neighbors": self._model.n_neighbors,
            "weights": self._model.weights,
            "p": self._model.p,
            "metric": self._model.metric,
        }

    def predict(
            self,
            observations: np.ndarray
        ) -> np.ndarray:
        """
        Predict using the KNearestNeighbors model.

        Args:
            observations (np.ndarray):
            Observations to predict.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self._model.predict(observations)
