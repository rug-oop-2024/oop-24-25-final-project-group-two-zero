import numpy as np
from sklearn.linear_model import LinearRegression
from .. import Model

class LinearRegressionModel(Model):
    """
    A wrapper around scikit-learn's LinearRegression model.
    """
    type = "regression"

    def __init__(self, fit_intercept=True, copy_X=True, n_jobs=None, **kwargs):
        """
        Initialize the Linear Regression model with hyperparameters.

        Args:
            fit_intercept (bool): Whether to calculate the intercept.
            copy_X (bool): If True, X will be copied; else, it may be overwritten.
            n_jobs (int): Number of jobs to use for computation.
        """
        super().__init__(**kwargs)
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self._model = LinearRegression(fit_intercept=self.fit_intercept, copy_X=self.copy_X, n_jobs=self.n_jobs)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        """
        Fit the Linear Regression model.

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
        Predict using the Linear Regression model.

        Args:
            observations (np.ndarray): Observations to predict.

        Returns:
            np.ndarray: Predicted values.
        """
        observations = np.asarray(observations)
        return self._model.predict(observations)
