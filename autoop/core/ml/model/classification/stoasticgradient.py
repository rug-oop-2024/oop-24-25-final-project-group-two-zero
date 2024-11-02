from .. import Model
import numpy as np
from sklearn.linear_model import SGDClassifier

class StochasticGradient(Model):
    type = "classification"

    def __init__(self, loss='hinge', penalty='l2', alpha=0.0001, max_iter=1000, **kwargs):
        '''
        Initialize the Stochastic Gradient Descent classifier with hyperparameters.

        Args:
            loss (str): Loss function ('hinge', 'log_loss', 'modified_huber', etc.).
            penalty (str): Penalty ('l2', 'l1', 'elasticnet').
            alpha (float): Regularization term.
            max_iter (int): Maximum number of iterations.
        '''
        super().__init__(**kwargs)
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self._model = SGDClassifier(loss=self.loss, penalty=self.penalty, alpha=self.alpha, max_iter=self.max_iter)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        '''
        Fit the Stochastic Gradient Descent model.

        Args:
            observations (np.ndarray): Training data features.
            ground_truth (np.ndarray): Training data labels.
        '''
        observations = np.asarray(observations)
        ground_truth = np.asarray(ground_truth)
        self._model.fit(observations, ground_truth)
        self._parameters['coef_'] = self._model.coef_
        self._parameters['intercept_'] = self._model.intercept_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        '''
        Predict using the Stochastic Gradient Descent model.

        Args:
            observations (np.ndarray): Observations to predict.

        Returns:
            np.ndarray: Predicted labels.
        '''
        observations = np.asarray(observations)
        return self._model.predict(observations)
