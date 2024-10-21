from .. import Model
import numpy as np
from sklearn.linear_model import SGDClassifier
from pydantic import Field, field_validator


class StoasticGradient(Model):
    loss_function: str = Field(default="hinge")
    penalty_function: str = Field(default="elasticnet")
    _model: SGDClassifier = SGDClassifier(loss = loss_function, penalty=penalty_function)

    @field_validator("loss_function")
    def validate_loss(cls, value):
        if value not in ["hinge","modified_huber", "log_log"]:
            raise ValueError("Value not in the list of loss is supported.")
        return value

    @field_validator("penalty_function")
    def validate_penalty(cls, value):
        if value not in ["l1", "l2", "elasticnet"]:
            raise ValueError("Value not in the list of penalty is supported.")
        return value

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        """
        Fit the StochasticGradient model using observations and ground truth.

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