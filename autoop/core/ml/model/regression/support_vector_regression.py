from sklearn.svm import SVR
from ..model import Model
import numpy as np
from typing import Any


class SupportVectorRegression(Model):
    """
    Support Vector Regression model.
    """

    _type: str = "regression"
    _available_hyperparameters: dict = {
        "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
        "degree": 3,
        "gamma": ["scale", "auto"],
        "coef0": 0.0,
        "tol": 1e-3,
        "C": 1.0,
        "epsilon": 0.1,
        "shrinking": True,
        "cache_size": 200,
        "verbose": False,
        "max_iter": -1,
    }
    _supported_feature_types: list = ["numerical"]
    _supported_target_types: list = ["numerical"]

    def __init__(self, **hyperparameters: Any) -> None:
        """
        Initializes the SupportVectorRegression model with hyperparameters.

        Args:
            **hyperparameters: Hyperparameters for the model.
        """
        super().__init__(**hyperparameters)
        # Merge default hyperparameters with user-provided ones
        params = {
            k: self._hyperparameters.get(k, v)
            for k, v in self._available_hyperparameters.items()
        }
        self._model = SVR(**params)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the model to the given data.

        Args:
            observations (np.ndarray): The input data to fit the model to.
            ground_truth (np.ndarray): The target values to fit the model to.
        """
        self._parameters = {
            "coef": self._model.coef_,
            "intercept": self._model.intercept_,
        }
        self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for the given observations.

        Args:
            observations (np.ndarray): The input data to predict.

        Returns:
            np.ndarray: The predicted target values.
        """
        return self._model.predict(observations)
