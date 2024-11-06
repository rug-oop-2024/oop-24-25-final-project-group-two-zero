# model.py
from abc import ABC, abstractmethod
import numpy as np
import copy
from typing import List


class Model(ABC):
    """
    Abstract base class for all models, containing fit and predict methods.
    """

    _type = None
    _available_hyperparameters = {}
    _supported_feature_types: List[str] = []
    _supported_target_types: List[str] = []

    def __init__(self, **hyperparameters) -> None:
        """
        Initializes the Model with hyperparameters.

        Args:
            **hyperparameters: Hyperparameters for the model.
        """
        self._parameters = {}
        self._hyperparameters = hyperparameters

    @property
    def type(self) -> str:
        return self._type

    @property
    def parameters(self) -> dict:
        return copy.deepcopy(self._parameters)

    @property
    def available_hyperparameters(self) -> dict:
        """
        Get the available hyperparameters and their default values.

        Returns:
            dict: A dictionary of hyperparameter names and default values.
        """
        return self._available_hyperparameters.copy()

    @property
    def supported_feature_types(self) -> List[str]:
        """
        Get the list of supported feature types.

        Returns:
            List[str]: Supported feature types.
        """
        return self._supported_feature_types

    @property
    def supported_target_types(self) -> List[str]:
        """
        Get the list of supported target types.

        Returns:
            List[str]: Supported target types.
        """
        return self._supported_target_types

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the model to the data.

        Args:
            observations (np.ndarray): Features.
            ground_truth (np.ndarray): Target values.
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Makes predictions based on observations.

        Args:
            observations (np.ndarray): Features for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        pass
