from .. import Model
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pydantic import Field, field_validator

class TreeClassification(Model):
    """
    Decision Tree Classifier
    """
    _model: DecisionTreeClassifier = DecisionTreeClassifier()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        """
        Fit the DecisionTree model using observations and ground truth.

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
        self._parameters = {'coef_': self._model.coef_, 'intercept_': self._model.intercept_}

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict using the DecisionTree model.

        Args:
            observations (np.ndarray): Observations with shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted targets with shape (n_samples,)
        """
        
        return self._model.predict(observations)

