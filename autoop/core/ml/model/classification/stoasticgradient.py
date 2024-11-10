from sklearn.linear_model import SGDClassifier
from ..model import Model
import numpy as np
from typing import Any


class StochasticGradient(Model):
    """Stochastic Gradient Descent (SGD) classifier."""

    _type: str = "classification"
    _available_hyperparameters: dict = {
        "loss": ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"],
        "penalty": ["l2", "l1", "elasticnet"],
        "alpha": 0.0001,
        "max_iter": 1000,
        "tol": 1e-3,
        "learning_rate": ["optimal", "constant", "invscaling", "adaptive"],
        "eta0": 0.0,
    }
    supported_feature_types: list = ["numerical"]
    supported_target_types: list = ["categorical"]

    def __init__(self: "StochasticGradient", **hyperparameters: Any) -> None:
        """
        Initialize the StochasticGradient model with hyperparameters.

        Args:
            **hyperparameters: Hyperparameters for the model.
        """

        super().__init__(**hyperparameters)
        self._params: dict = {
            k: self._hyperparameters.get(k, v)
            for k, v in self._available_hyperparameters.items()
        }
        # Make only the hyperparameters that are chosen by the user
        self._model: SGDClassifier = SGDClassifier(**self._params)

    def fit(
        self: "StochasticGradient", observations: np.ndarray, ground_truth: np.ndarray
    ) -> None:
        """
        Fit the model to the data.

        Args:
            observations (np.ndarray): Features.
            ground_truth (np.ndarray): Target values.
        """
        self._model.fit(observations, ground_truth)
        self._parameters: dict = {
            "coef_": self._model.coef_,
            "intercept_": self._model.intercept_,
            "n_iter_": self._model.n_iter_,
            "classes_": self._model.classes_,
        }

    def predict(self: "StochasticGradient", observations: np.ndarray) -> np.ndarray:
        """
        Predict using the StochasticGradient model.

        Args:
            observations (np.ndarray):
            Observations to predict.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self._model.predict(observations)
