from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    """
    Abstract base class for all models, containing fit and predict methods.
    """
    _type  = None
    def __init__(self) -> None:
        """
        Initializes the Model with an empty parameters dictionary.
        """
        self.parameters = {}

    @abstractmethod
    def fit(self, observations: np.ndarray, groundtruth: np.ndarray) -> None:
        """
        Fits the model to the data.

        Args:
            observations (np.ndarray): Features.
            groundtruth (np.ndarray): Target values.
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

    @property
    def type(self) -> str:
        return self._type
