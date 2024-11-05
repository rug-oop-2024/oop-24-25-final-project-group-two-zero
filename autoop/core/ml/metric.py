from abc import ABC, abstractmethod
import numpy as np
from typing import Any

METRICS = [
    "MeanSquaredError",  # Regression tasks
    "accuracy",          # Classification tasks
    "mean_absolute_error",
    "F_one_score",
    "specificity",
    "r_squared_error",
]


def get_metric(name: str) -> Any:
    """
    Retrieve a metric class by its name.

    Args:
        name (str): The name of the metric to retrieve.

    Returns:
        Any: The metric class corresponding to the given name.

    Raises:
        ValueError: If the metric with the specified name does not exist.
    """
    if name not in METRICS:
        raise ValueError(f"Metric {name} does not exist")
    return getattr(__import__("autoop.core.ml.metric", fromlist=[name]), name)


class Metric(ABC):
    """Base class for all metrics."""
    _name = None

    @abstractmethod
    def evaluate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute the metric."""
        pass

    @property
    def name(self) -> str:
        return self._name


class MeanSquaredError(Metric):
    _name = "Mean Squared Error"

    def evaluate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)


class MeanAbsoluteError(Metric):
    _name = "Mean Absolute Error"

    def evaluate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))


class R2Score(Metric):
    _name = "R-Squared"

    def evaluate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)


class Accuracy(Metric):
    _name="Accuracy"

    def evaluate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(y_true == y_pred)


class Specificity(Metric):
    _name="Specificity"

    def evaluate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        true_negative = np.sum((y_true == 0) & (y_pred == 0))
        false_positive = np.sum((y_true == 0) & (y_pred == 1))
        if (true_negative + false_positive) == 0:
            return 0.0
        return true_negative / (true_negative + false_positive)


class F1Score(Metric):
    _name="F1 Score"

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
