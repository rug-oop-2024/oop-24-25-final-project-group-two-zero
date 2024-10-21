from .. import Model
import numpy as np
from sklearn.linear_model import Ridge
from pydantic import Field, field_validator

class LogisticRegression(Model):
    alpha: float = Field(default=0.5, ge=0, le=1)
    
    _model: Ridge = Ridge(alpha)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        """
        Fit the LogisticRegression model using observations and ground truth.

        Args:
            observations (np.ndarray): Observations with shape (n_samples, n_features)
            ground_truth (np.ndarray): Ground truth targets with shape (n_samples,)

        Returns:
            None

        Stores:
            self._parameters (dict): Contains the model parameters.
        """
        observations = np.asarray(observations)
        ground_truth = np.asarray(ground_truth)

        self._model.fit(observations, ground_truth)
        self._parameters = {
            'coef_': self._model.coef_,
            'intercept_': self._model.intercept_,
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict using the LogisticRegression model.

        Args:
            observations (np.ndarray): Observations with shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted targets with shape (n_samples,)
        """
        if self._parameters is None:
            raise ValueError("Model has not been fitted yet. Please call 'fit' before 'predict'.")
        observations = np.asarray(observations)
        return self._model.predict(observations)