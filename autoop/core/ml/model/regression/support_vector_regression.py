from sklearn.svm import SVR
from ..model import Model
import numpy as np
from typing import Optional, Dict, List


class SupportVectorRegression(Model):
    """
    Support Vector Regression model.

    Contains a type, available hyperparameters,
    supported feature types, and target types.
    """

    _type: str = "regression"
    _available_hyperparameters: dict = {
        "kernel": [
            "linear",
            "poly",
            "rbf",
            "sigmoid",
            "precomputed",
        ],
        "degree": 3,
        "gamma": ["scale", "auto"],
        "coef0": 0.0,
        "tol": 1e-3,
        "C": 1.0,
        "epsilon": 0.1,
        "shrinking": True,
        "cache_size": 200,
        "verbose": False,
        "max_iter": 1000,
    }
    _supported_feature_types: List[str] = ["numerical"]
    _supported_target_types: List[str] = ["numerical"]

    def __init__(
        self: "SupportVectorRegression",
        **hyperparameters: Optional[Dict]
    ) -> None:
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

    def fit(
        self: "SupportVectorRegression",
        observations: np.ndarray,
        ground_truth: np.ndarray,
    ) -> None:
        """
        Fits the model to the given data.

        Args:
            observations (np.ndarray): The input data to fit the model to.
            ground_truth (np.ndarray): The target values to fit the model to.
        """
        if observations.ndim != 2:
            raise ValueError("Observations must be a 2-dimensional array")
        if ground_truth.ndim != 1:
            raise ValueError("Ground truth must be a 1-dimensional array")
        if observations.shape[0] != ground_truth.shape[0]:
            raise ValueError("Number observations != ground truth values")
        if observations.shape[0] == 0:
            raise ValueError("Cannot fit model with empty dataset")
        self._model.fit(observations, ground_truth)
        self._parameters = {
            "support_": self._model.support_,
            "support_vectors_": self._model.support_vectors_,
            "dual_coef_": self._model.dual_coef_,
            "intercept_": self._model.intercept_,
        }

    def predict(
        self: "SupportVectorRegression",
        observations: np.ndarray
    ) -> np.ndarray:
        """
        Predicts the target values for the given observations.

        Args:
            observations (np.ndarray): The input data to predict.

        Returns:
            np.ndarray: The predicted target values.
        """
        return self._model.predict(observations)
