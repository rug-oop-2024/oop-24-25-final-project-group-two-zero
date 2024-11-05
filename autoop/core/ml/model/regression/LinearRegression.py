# linear_regression.py
from sklearn.linear_model import LinearRegression
from .model import Model
import numpy as np


class LinearRegressionModel(Model):
    """
    Linear Regression model.
    """

    _type = "regression"
    _available_hyperparameters = {
        'fit_intercept': True,
        'copy_X': True,
        'n_jobs': None,
        'positive': False,
    }

    def __init__(self, **hyperparameters) -> None:
        """
        Initializes the LinearRegressionModel model with hyperparameters.

        Args:
            **hyperparameters: Hyperparameters for the model.
        """
        super().__init__(**hyperparameters)
        params = {k: self._hyperparameters.get(k, v) for k, v in self._available_hyperparameters.items()}
        self._model = LinearRegression(**params)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the linear regression model to the provided training data.

        Args:
            observations (np.ndarray): Training data features.
            ground_truth (np.ndarray): Training data targets.
        """
        self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict using the linear regression model.

        Args:
            observations (np.ndarray): Observations to predict.

        Returns:
            np.ndarray: Predicted values.
        """
        return self._model.predict(observations)
