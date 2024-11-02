from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy

class Model(ABC):

    def __init__(self) -> None:
        self.parameters = {}  # Initialize as a standard dictionary


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
