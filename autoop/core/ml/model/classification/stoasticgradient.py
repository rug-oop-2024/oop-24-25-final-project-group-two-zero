# sgd.py
from sklearn.linear_model import SGDClassifier
from ..model import Model
import numpy as np


class StochasticGradient(Model):
    """
    Stochastic Gradient Descent (SGD) classifier.
    """

    _type = "classification"
    _available_hyperparameters = {
        'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': 0.0001,
        'max_iter': 1000,
        'tol': 1e-3,
        'learning_rate': ['optimal', 'constant', 'invscaling', 'adaptive'],
        'eta0': 0.0,
    }

    def __init__(self, **hyperparameters) -> None:
        """
        Initializes the StochasticGradient model with hyperparameters.

        Args:
            **hyperparameters: Hyperparameters for the model.
        """

        super().__init__(**hyperparameters)
        params = {k: self._hyperparameters.get(k, v) for k, v in self._available_hyperparameters.items()}
        self._model = SGDClassifier(**params)

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
        Predict using the StochasticGradient model.

        Args:
            observations (np.ndarray): Observations to predict.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self._model.predict(observations)
