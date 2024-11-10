from sklearn.linear_model import LinearRegression
from ..model import Model
import numpy as np
from typing import Any


class MultipleLinearRegression(Model):
    """Linear Regression model."""

    _type: str = "regression"
    _available_hyperparameters: dict = {
        "fit_intercept": True,
        "copy_X": True,
        "n_jobs": None,
        "positive": False,
    }
    _supported_feature_types: list = ["numerical"]
    _supported_target_types: list = ["numerical"]

    def __init__(
        self: "MultipleLinearRegression",
        **hyperparameters: Any
    ) -> None:
        """
        Initializes the LinearRegressionModel
        model with hyperparameters.

        Args:
            **hyperparameters:
            Hyperparameters for the model.
        """
        super().__init__(**hyperparameters)
        self.user_choices = {
            k: self._hyperparameters.get(k, v)
            for k, v in self._available_hyperparameters.items()
        }
        self._model = LinearRegression(**self.user_choices)

    def fit(
        self: "MultipleLinearRegression",
        observations: np.ndarray,
        ground_truth: np.ndarray,
    ) -> None:
        """
        Fits the linear regression model provided training data.

        Args:
            observations (np.ndarray):
                Training data features.
            ground_truth (np.ndarray):
                Training data targets.
        """
        self._model.fit(observations, ground_truth)
        self._parameters = {
            "intercept_": self._model.intercept_,
            "coef_": self._model.coef_,
        }

    def predict(
        self: "MultipleLinearRegression", observations: np.ndarray
    ) -> np.ndarray:
        """
        Predict using the linear regression model.

        Args:
            observations (np.ndarray): Observations to predict.

        Returns:
            np.ndarray: Predicted values.
        """
        return self._model.predict(observations)
