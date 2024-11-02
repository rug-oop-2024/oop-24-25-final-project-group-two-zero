
# metric.py
from abc import ABC, abstractmethod
from typing import Any, Callable
import numpy as np
from math import sqrt
# Put the list in a list and then do the class for making the choices and 
# Putting them like Mo shakoush said


METRICS = [
    "MeanSquaredError", # Regression tasks
    "accuracy", # Classification tasks
    "mean_absolute_error",
    "F_one_score",
    "specificity"
    "r_squared_error",
]

def get_metric(name: str):
    if name not in METRICS:
        raise ValueError(f"Metric {name} does not exist")
    return getattr(__import__("autoop.core.ml.metric", fromlist=[name]), name)


# metric.py
from abc import ABC, abstractmethod
import numpy as np

class Metric(ABC):
    """Base class for all metrics."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute the metric."""
        pass



# add here concrete implementations of the Metric class

# metric.py (continued)

class MeanSquaredError(Metric):
    def __init__(self):
        super().__init__(name="Mean Squared Error")

    def evaluate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

class MeanAbsoluteError(Metric):
    def __init__(self):
        super().__init__(name="Mean Absolute Error")

    def evaluate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))

class R2Score(Metric):
    def __init__(self):
        super().__init__(name="R-Squared")

    def evaluate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

class Accuracy(Metric):
    def __init__(self):
        super().__init__(name="Accuracy")

    def evaluate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(y_true == y_pred)

class Specificity(Metric):
    def __init__(self):
        super().__init__(name="Specificity")

    def evaluate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        true_negative = np.sum((y_true == 0) & (y_pred == 0))
        false_positive = np.sum((y_true == 0) & (y_pred == 1))
        if (true_negative + false_positive) == 0:
            return 0.0
        return true_negative / (true_negative + false_positive)

class F1Score(Metric):
    def __init__(self):
        super().__init__(name="F1 Score")

    def evaluate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        true_positive = np.sum((y_true == 1) & (y_pred == 1))
        false_positive = np.sum((y_true == 0) & (y_pred == 1))
        false_negative = np.sum((y_true == 1) & (y_pred == 0))

        if (true_positive + false_positive) == 0 or (true_positive + false_negative) == 0:
            return 0.0

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)

        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
