import numpy as np
from .. import Model

class MultipleLinearRegression(Model):
    """
    Multiple Linear Regression model implemented from scratch.
    """
    type = "regression"

    def __init__(self, fit_intercept=True, **kwargs):
        """
        Initialize the multiple linear regression model.

        Args:
            fit_intercept (bool): Whether to calculate the intercept term.
        """
        super().__init__(**kwargs)
        self.fit_intercept = fit_intercept

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the multiple linear regression model using the normal equation.

        Args:
            observations (np.ndarray): Training data features.
            ground_truth (np.ndarray): Training data targets.
        """
        observations = np.asarray(observations)
        ground_truth = np.asarray(ground_truth)
        if observations.shape[0] != ground_truth.shape[0]:
            raise ValueError("Number of samples in observations and ground_truth must be equal.")
        if self.fit_intercept:
            X_design = np.hstack((observations, np.ones((observations.shape[0], 1))))
        else:
            X_design = observations
        XtX = X_design.T @ X_design
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)
        w = XtX_inv @ X_design.T @ ground_truth
        self._parameters['weights'] = w

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict using the linear model.

        Args:
            observations (np.ndarray): Observations to predict.

        Returns:
            np.ndarray: Predicted values.
        """
        if 'weights' not in self._parameters:
            raise ValueError("Model is not fitted yet. Please call 'fit' before 'predict'.")
        observations = np.asarray(observations)
        if self.fit_intercept:
            X_design = np.hstack((observations, np.ones((observations.shape[0], 1))))
        else:
            X_design = observations
        w = self._parameters['weights']
        return X_design @ w
