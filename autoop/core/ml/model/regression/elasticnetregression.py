from typing import Literal
import numpy as np
from .. import Model
class ElasticNetRegression(Model):
    """
    ElasticNet Regression model.
    """

    def __init__(
        self,
        num_iterations: int = 1000,
        alpha: float = 0.01,
        l1_ratio: float = 0.5,
        l2_ratio: float = 0.5
        ) -> None:
        super().__init__()
        self.theta = None
        self._numiterations = num_iterations
        self._alpha = alpha
        self._l1_ratio = l1_ratio
        self._l2_ratio = l2_ratio


    def innit_theta(self, observations: np.ndarray) -> np.ndarray:
        m,n = observations.shape
        self.theta = np.zeros((n,1))
        return m
    
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        """
        Residuals for elastic net regression

        Args:
            observations (np.ndarray): Observations (n_samples, n_features)
            ground_truth (np.ndarray): Ground truth (n_samples, )

        Returns:
            np.ndarray: Residuals
        """
        observations: np.ndarray = np.asarray(observations)
        ground_truth: np.ndarray = np.asarray(ground_truth)

        m, n = observations.shape
        o, p = ground_truth.shape
        if m != p:
            raise ValueError("Number of samples in observations and ground_truth must be equal.")
        
        m = self.innit_theta(observations)

        for _ in range(self._numiterations):
            predictions = observations @ self.theta
            errors = predictions - ground_truth
            # This is the gradient without any regularization
            gradient = (2/m) * observations.T @ errors
            # This is the gradient with l1 regularization
            gradient += 2 * self._l1_ratio * self.theta
            # This is the gradient with l2 regularization
            gradient += self._l2_ratio * np.sign(self.theta)
            theta -= self._alpha * gradient

        self._parameters['fitted_values'] = predictions

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict using the linear model.

        Args:
            observations (np.ndarray): Observations (n_samples, n_features)

        Returns:
            np.ndarray: Predicted values (n_samples, )
        """
        # Check if model is fitted and is a nice check
        if 'fitted_values' not in self._parameters:
            raise ValueError("""Model is not fitted yet.
                            Please call 'fit' before 'predict'.""")
        # This is to ensure that observations is a numpy array
        observations: np.ndarray = np.asarray(observations)
        return observations @ self.theta