from pydantic import BaseModel
from abc import abstractmethod, ABC
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal

class Model(ABC):

    def __init__(self, type: Literal["regression", "classification"]):
        self.type = type
        self._parameters = {}


    @property
    def get_parameters(self):
        return deepcopy(self._parameters)


    @get_parameters.setter
    def set_parameters(self, parameters):
        self._parameters = parameters


    @abstractmethod
    def fit(self, observations: np.ndarray, groundtruth: np.ndarray) -> None:
        """
        Fits the model to the data
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Target
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Makes predictions
        Args:
            observations (np.ndarray): Features
        Returns:
            np.ndarray: Predictions
        """
        pass

    
