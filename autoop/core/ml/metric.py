
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


class Metric(ABC):
    """Base class for all metrics.
    """
    # your code here
    # remember: metrics take ground truth and prediction as input and return a real number
    @abstractmethod
    def doing_math(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    def __call__(self,function: Callable):
        raise NotImplementedError("To be implemented.")
    


# add here concrete implementations of the Metric class

class MeanSquaredError(Metric):
    # This is a regression Metric
    def doing_math(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_pred) != len(y_true):
            raise ValueError("y_pred and y_true should have the same length")
        return np.mean((y_true - y_pred)**2)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.doing_math(y_true, y_pred)

class mean_absolute_error(Metric):
    # This is a regression Metric
    def doing_math(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:

        if len(y_pred) != len(y_true):
            raise ValueError("y_pred and y_true should have the same length")
        return np.mean(abs(y_true - y_pred))

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.doing_math(y_true, y_pred)

class r_squared_error(Metric):
    # This is a regression Metric
    def doing_math(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:

        if len(y_pred) != len(y_true):
            raise ValueError("y_pred and y_true should have the same length")
        return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.doing_math(y_true, y_pred)

# This is the classification Metric

class accuracy(Metric):
    # This is a classification Metric
    def doing_math(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:

        if len(y_pred) != len(y_true):
            raise ValueError("y_pred and y_true should have the same length")
        _count = 0

        for i in range(len(y_pred)):
            if y_pred[i] != y_true[i]:
                _count += 1
        return _count / len(y_pred)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.doing_math(y_true, y_pred)

class specificity(Metric):
    # This is a classification Metric
    def doing_math(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_pred) != len(y_true):
            raise ValueError("y_pred and y_true should have the same length")

        true_negative = 0
        false_positve = 0

        for true, pred in zip(y_true, y_pred):
            if true == 0 and pred == 0:
                true_negative += 1  # True Negative
            elif true == 0 and pred == 1:
                false_positve += 1  # False Positive

    # Calculate specificity
        if (true_negative + false_positve) == 0:
            return 0
        specificity = true_negative / (true_negative + false_positve)
        return specificity

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.doing_math(y_true, y_pred)

class F_one_score(Metric):
    def doing_math(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_pred) != len(y_true):
            raise ValueError("y_pred and y_true should have the same length")
            # Initialize counts
        true_positive = 0
        false_positive = 0
        false_negative = 0
    
        for true, pred in zip(y_true, y_pred):
            if true == 1 and pred == 1:
                true_positive += 1  # True Positive
            elif true == 0 and pred == 1:
                false_positive += 1  # False Positive
            elif true == 1 and pred == 0:
                false_negative += 1  # False Negative

        # Calculate precision and recall

        if (true_positive + false_positive) == 0:
            return 0
        else:
            precision = true_positive / (true_positive + false_positive)

        if (true_positive + false_negative) == 0:
            return 0
        else:
            recall = true_positive / (true_positive + false_negative)


        if precision + recall == 0:
            return 0
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.doing_math(y_true, y_pred)
