from .. import Model
import numpy as np
from sklearn.linear_model import Ridge

class RidgeRegression(Model):
    type = "regression"

    def __init__(self, alpha=1.0, fit_intercept=True, solver='auto', **kwargs) -> None:
        '''
        Initialize the Ridge Regression model with hyperparameters.

        Args:
            alpha (float): Regularization strength.
            fit_intercept (bool): Whether to calculate the intercept.
            solver (str): Solver to use (`'auto'`, `'svd'`, `'cholesky'`, etc.).
        '''
        super().__init__(**kwargs)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self._model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept, solver=self.solver)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the Ridge Regression model.

        Args:
            observations (np.ndarray): Training data features.
            ground_truth (np.ndarray): Training data targets.
        """
        observations = np.asarray(observations)
        ground_truth = np.asarray(ground_truth)
        self._model.fit(observations, ground_truth)
        self.parameters['coef_'] = self._model.coef_
        self.parameters['intercept_'] = self._model.intercept_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict using the Ridge Regression model.

        Args:
            observations (np.ndarray): Observations to predict.

        Returns:
            np.ndarray: Predicted values.
        """
        observations = np.asarray(observations)
        return self._model.predict(observations)
