import numpy as np
from .. import Model


class MultipleLinearRegression(Model):
    """
    Multiple Linear Regression model.
    """
    def __init__(self) -> None:
        """
        This initializes the multiple linear regression model.
        """
        super().__init__()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the multiple linear regression model using the normal equation.

        Args:
            observations (np.ndarray): Observations (n_samples, n_features)
            ground_truth (np.ndarray): Ground truth (n_samples, )

        Returns:
            None

        Stores:
            self._parameters['fitted_values'] (np.ndarray): Coefficient vector
            including intercept term
        """
        # Ensure observations and ground_truth are numpy arrays
        observations: np.ndarray = np.asarray(observations)
        ground_truth: np.ndarray = np.asarray(ground_truth)

        # Check dimensions
        if observations.shape[0] != ground_truth.shape[0]:
            raise ValueError("""Number of samples in
                     observations and ground_truth must be equal.""")

        # Add a column of ones to observations for the intercept term
        n_samples: int = observations.shape[0]
        X_bias: np.ndarray = np.hstack((observations, np.ones((n_samples, 1))))

        # This is the multiplication of the transpose of X_bias * X_bias
        XtX: np.ndarray = np.dot(X_bias.T, X_bias)
        try:
            XtX_inv: np.ndarray = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:

            # Use pseudo-inverse if XtX is singular
            XtX_inv: np.ndarray = np.linalg.pinv(XtX)

        # This is the continuation of the formula in the slides
        w: np.ndarray = XtX_inv @ X_bias.T @ ground_truth

        # Store fitted values in self._parameters
        self._parameters['fitted_values'] = w

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
        # Add a column of ones to observations for the intercept term
        n_samples: int = observations.shape[0]
        X_bias: np.ndarray = np.hstack((observations, np.ones((n_samples, 1))))
        # This takes the parameter "parameters" from self._parameters
        w: np.ndarray = self._parameters['fitted_values']
        y_pred: np.ndarray = np.dot(X_bias, w)
        return y_pred