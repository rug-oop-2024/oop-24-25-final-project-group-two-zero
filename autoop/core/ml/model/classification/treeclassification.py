from sklearn.tree import DecisionTreeClassifier
from ..model import Model
import numpy as np
from typing import Any


class TreeClassification(Model):
    """Decision Tree Classifier."""

    _type = "classification"
    _available_hyperparameters = {
        "criterion": ["gini", "entropy", "log_loss"],
        "splitter": ["best", "random"],
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": None,
    }

    supported_feature_types = ["numerical"]
    supported_target_types = ["categorical"]

    def __init__(self: "TreeClassification", **hyperparameters: Any) -> None:
        """
        Initializes the TreeClassification model
        with hyperparameters.

        Args:
            **hyperparameters: Hyperparameters
            for configuring the DecisionTreeClassifier.
        """
        super().__init__(**hyperparameters)
        self._params = {
            k: self._hyperparameters.get(k, v)
            for k, v in self._available_hyperparameters.items()
        }
        self._model = DecisionTreeClassifier(**self._params)

    def fit(
        self: "TreeClassification", observations: np.ndarray, ground_truth: np.ndarray
    ) -> None:
        """
        Fits the model to the data.

        Args:
            observations (np.ndarray): Features.
            ground_truth (np.ndarray): Target values.
        """
        self._model.fit(observations, ground_truth)
        self._parameters = {
            "criterion": self._model.criterion,
            "splitter": self._model.splitter,
            "max_depth": self._model.max_depth,
            "min_samples_split": self._model.min_samples_split,
            "min_samples_leaf": self._model.min_samples_leaf,
            "max_features": self._model.max_features,
            "n_classes_": self._model.n_classes_,
            "n_features_in_": self._model.n_features_in_,
            "feature_importances_": self._model.feature_importances_,
            "classes_": self._model.classes_,
        }

    def predict(self: "TreeClassification", observations: np.ndarray) -> np.ndarray:
        """
        Predict using the TreeClassification model.

        Args:
            observations (np.ndarray): Observations to predict.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self._model.predict(observations)
