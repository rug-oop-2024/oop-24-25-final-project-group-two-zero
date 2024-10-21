from abc import ABC, abstractmethod
from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error", # Regression tasks
    "accuracy", # Classification tasks
] # add the names (in strings) of the metrics you implement

def get_metric(name: str):
    if name not in METRICS:
        raise ValueError(f"Metric {name} does not exist")
    return getattr(__import__("autoop.core.ml.metric", fromlist=[name]), name)


class Metric(ABC):
    """Base class for all metrics.
    """
    # your code here
    # remember: metrics take ground truth and prediction as input and return a real number
    @abstractmethod
    def math(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    def __call__(self):
        raise NotImplementedError("To be implemented.")
    


# add here concrete implementations of the Metric class
class accuracy(Metric):

    def math(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:

        if len(y_pred) != len(y_true):
            raise ValueError("y_pred and y_true should have the same length")
        _count = 0

        for i in range(len(y_pred)):
            if y_pred[i] != y_true[i]:
                _count += 1
        return _count / len(y_pred)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.math(y_true, y_pred)
class mean_squared_error(Metric):

    def math(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:

        if len(y_pred) != len(y_true):
            raise ValueError("y_pred and y_true should have the same length")
        return np.mean((y_true - y_pred)**2)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.math(y_true, y_pred)
    