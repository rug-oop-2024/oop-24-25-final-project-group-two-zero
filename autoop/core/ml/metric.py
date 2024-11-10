from abc import ABC, abstractmethod
import numpy as np
from typing import Any

METRICS = [
    "MeanSquaredError",  # Regression tasks
    "Accuracy",  # Classification tasks
    "MeanAbsoluteError",
    "F1Score",
    "Specificity",
    "R2Score",
]


def get_metric(name: str) -> Any:
    """
    Retrieve a metric class by its name.

    Args:
        name (str): The name of the metric to retrieve.

    Returns:
        Any: The metric class corresponding to the given name.

    Raises:
        ValueError: If the metric with
            the specified name does not exist.
    """
    if name not in METRICS:
        raise ValueError(
            f"""Metric
                     {name}does not exist"""
        )
    return getattr(__import__("autoop.core.ml.metric", fromlist=[name]), name)


class Metric(ABC):
    """Base class for all metrics."""

    _name: str = None

    @abstractmethod
    def evaluate(
        self: "Metric",
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """Compute the metric."""
        pass

    @property
    def name(self:'Metric') -> str:
        """
        Get the name of the metric.

        Returns:
            str: The name of the metric.
        """
        return self._name


class MeanSquaredError(Metric):
    """Mean Squared Error metric."""

    _name: str = "Mean Squared Error"

    def evaluate(
        self: "MeanSquaredError",
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """
        Compute the mean squared error.

        between the predictions and the true labels.

        Args:
            y_pred (np.ndarray): The predictions.
            y_true (np.ndarray): The true labels.

        Returns:
            float: The mean squared error.
        """
        return np.mean((y_true - y_pred) ** 2)


class MeanAbsoluteError(Metric):
    """Mean Absolute Error metric."""

    _name: str = "Mean Absolute Error"

    def evaluate(
        self: "MeanAbsoluteError", y_pred: np.ndarray, y_true: np.ndarray
    ) -> float:
        """
        Compute the mean absolute error
        between the predictions and the true labels.

        Args:
            y_pred (np.ndarray): The predictions.
            y_true (np.ndarray): The true labels.

        Returns:
            float: The mean absolute error.
        """
        return np.mean(np.abs(y_true - y_pred))


class R2Score(Metric):
    """R-Squared metric."""

    _name: str = "R-Squared"

    def evaluate(
        self: "R2Score",
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """
        Compute the R-squared score
        between the predictions and the true labels.

        Args:
            y_pred (np.ndarray): The predictions.
            y_true (np.ndarray): The true labels.

        Returns:
            float: The R-squared score.
        """
        ss_res: float = np.sum((y_true - y_pred) ** 2)
        ss_tot: float = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)


class Accuracy(Metric):
    """Accuracy metric."""

    _name = "Accuracy"

    def evaluate(self: "Accuracy", y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the accuracy between the predictions and the true labels.

        Args:
            y_pred (np.ndarray): The predictions.
            y_true (np.ndarray): The true labels.

        Returns:
            float: The accuracy.
        """
        return np.mean(y_true == y_pred)


class Specificity(Metric):
    """Specificity metric."""

    _name = "Specificity"

    def evaluate(self: "Specificity", y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the specificity between
        the predictions and the true labels.

        The specificity is the number of
        true negatives divided by the sum of true
        negatives and false positives.

        Args:
            y_pred (np.ndarray): The predictions.
            y_true (np.ndarray): The true labels.

        Returns:
            float: The specificity.
        """
        true_negative = np.sum((y_true == 0) & (y_pred == 0))
        false_positive = np.sum((y_true == 0) & (y_pred == 1))
        if (true_negative + false_positive) == 0:
            return 0.0
        return true_negative / (true_negative + false_positive)


class F1Score(Metric):
    """F1 Score metric."""

    _name = "F1 Score"

    def evaluate(self: "F1Score", y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the F1 score between
        the predictions and the true labels.

        The F1 score is the harmonic mean of
        precision and recall and is a measure of
        the accuracy of a test. It is a single
        number that balances between precision and
        recall.

        Args:
            y_pred (np.ndarray): The predictions.
            y_true (np.ndarray): The true labels.

        Returns:
            float: The F1 score.
        """
        true_positive: float = np.sum((y_true == 1) & (y_pred == 1))
        false_positive: float = np.sum((y_true == 0) & (y_pred == 1))
        false_negative: float = np.sum((y_true == 1) & (y_pred == 0))

        if (true_positive + false_positive) == 0 or (
            true_positive + false_negative
        ) == 0:
            return 0.0

        precision: float = true_positive / (true_positive + false_positive)
        recall: float = true_positive / (true_positive + false_negative)

        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
