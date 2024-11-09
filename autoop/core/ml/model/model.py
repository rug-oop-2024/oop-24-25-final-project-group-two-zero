# model.py
from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy
from typing import List, Dict, Any


class Model(ABC):
    """
    Abstract base class for all models, containing fit and predict methods.
    """

    _type: str|None = None
    _available_hyperparameters: dict = {}
    _supported_feature_types: List[str] = []
    _supported_target_types: List[str] = []


    def __init__(self, **hyperparameters) -> None:
        """
        Initializes the Model with hyperparameters.

        Args:
            **hyperparameters: Hyperparameters for the model.
        """
        self._parameters: dict = {}
        self._hyperparameters: dict = hyperparameters

    @property
    def type(self) -> str:
        """
        Get the type of the model.

        Returns:
            str: The type of the model.
        """
        return self._type

    @property
    def parameters(self) -> dict:
        """
        Get the hyperparameters of the model.

        Returns:
            dict: A dictionary of hyperparameter names and values.
        """
        return deepcopy(self._parameters)

    @property
    def available_hyperparameters(self) -> dict:
        """
        Get the available hyperparameters and their default values.

        Returns:
            dict: A dictionary of hyperparameter names and default values.
        """
        return deepcopy(self._available_hyperparameters)

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

    def get_estimator(self):
        """
        Returns the underlying scikit-learn estimator.

        Returns:
            estimator: The underlying estimator object.
        """
        return self._model

    def get_hyperparameter_space(self, acceptable_ranges: Dict[str, any]) -> Dict[str, any]:
        """
        Returns the hyperparameter grid for tuning.

        Args:
            acceptable_ranges (dict): Acceptable ranges or options for hyperparameters.

        Returns:
            dict: Hyperparameter grid for tuning.
        """
        param_grid: Dict[str, any] = {}
        for param, value in acceptable_ranges.items():
            if isinstance(value, list):
                param_grid[param] = value
            elif isinstance(value, tuple):
                # Generate a list of values within the range
                if isinstance(value[0], int) and isinstance(value[1], int):
                    param_grid[param] = list(range(int(value[0]), int(value[1]) + 1))
                else:
                    # For floats, generate a list with reasonable steps
                    param_grid[param] = np.linspace(value[0], value[1], num=5).tolist()
            else:
                param_grid[param] = [value]
        return param_grid
    
    @property
    def supported_feature_types(self) -> List[str]:
        """
        Gets the supported feature types.

        Returns:
            List[str]: A list of supported feature types.
        """
        return deepcopy(self._supported_feature_types)

    @property
    def supported_target_types(self) -> List[str]:
        """
        Gets the supported target types.

        Returns:
            List[str]: A list of supported target types.
        """
        return deepcopy(self._supported_target_types)

    def set_params(self, **params) -> None:
        """
        Sets the parameters of the model.

        Args:
            **params: The parameters to set. Names must match the parameters in the model.
        """
        self._model.set_params(**params)
