from sklearn.linear_model import Ridge
from ..model import Model
import numpy as np


class RidgeRegression(Model):
    """
    Ridge Regression model.
    """

    _type: str = "regression"
    _available_hyperparameters: dict = {
        'alpha': 1.0,
        'fit_intercept': True,
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        'max_iter': None,
        'tol': 1e-3,
    }

    _supported_feature_types: list = ['numerical']
    _supported_target_types: list = ['numerical']

    def __init__(self, **hyperparameters) -> None:
        """
        Initializes the RidgeRegression model with specified hyperparameters.

        Args:
            **hyperparameters: Arbitrary keyword arguments for model hyperparameters.
        """
        super().__init__(**hyperparameters)
        self._parameters = {k: self._hyperparameters.get(k, v) for k, v in self._available_hyperparameters.items()}
        self._model = Ridge(**self._parameters)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the RidgeRegression model to the data.

        Args:
            observations (np.ndarray): Features.
            ground_truth (np.ndarray): Target values.
        """
        self._parameters = {
            "alpha": self._model.alpha_,
            "fit_intercept": self._model.fit_intercept_,
            "solver": self._model.solver_,
            "max_iter": self._model.max_iter_,
            "tol": self._model.tol_,
            "intercept": self._model.intercept_,
        }
        self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict using the RidgeRegression model.

        Args:
            observations (np.ndarray): Observations to predict.

        Returns:
            np.ndarray: Predicted values.
        """
        return self._model.predict(observations)
