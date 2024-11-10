from sklearn.linear_model import Ridge
from ..model import Model
import numpy as np
from typing import Any


class RidgeRegression(Model):
    """
    Ridge Regression model.
    """

    _type: str = "regression"
    _type: str = "regression"
    _available_hyperparameters: dict = {
        "alpha": 1.0,                     # Regularization strength (float)
        "fit_intercept": True,            # Whether to calculate intercept (boolean)
        "solver": [
            "auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"
        ],                                # Solver options (list of strings)
        "max_iter": None,                 # Maximum iterations for some solvers (int or None)
        "tol": 1e-3                       # Tolerance for stopping criteria (float)
    }
    _supported_feature_types: list = ["numerical"]
    _supported_target_types: list = ["numerical"]

    def __init__(self, **hyperparameters:Any) -> None:
        """
        Initializes the RidgeRegression model
        with specified hyperparameters.

        Args:
            **hyperparameters: Arbitrary keyword
                arguments for model hyperparameters.
        """
        super().__init__(**hyperparameters)
        self._params = {
            k: self._hyperparameters.get(k, v)
            for k, v in self._available_hyperparameters.items()
        }
        self._model = Ridge(**self._params)

    def fit(
            self,
            observations: np.ndarray,
            ground_truth: np.ndarray
        ) -> None:
        """
        Fits the RidgeRegression model to the data.

        Args:
            observations (np.ndarray): Features.
            ground_truth (np.ndarray): Target values.
        """
        self._model.fit(observations, ground_truth)
        self._parameters = {
            "intercept_": self._model.intercept_,
            "coef_": self._model.coef_,
        }

    def predict(
            self,
            observations: np.ndarray
        ) -> np.ndarray:
        """
        Predict using the RidgeRegression model.

        Args:
            observations (np.ndarray):
                Observations to predict.

        Returns:
            np.ndarray: Predicted values.
        """
        return self._model.predict(observations)
