import numpy as np
from collections import Counter
from pydantic import Field, field_validator
from .. import Model


class KNearestNeighbors(Model):
    '''
    This class is for K neighboring neighbors algorithm. It
    has the functions fit, predict, and _predict_single.
    '''
    k: int = Field(title="Number of neighbors", default=3)

    @field_validator("k")
    def k_greater_than_zero(cls, value : int) -> int|None:
        '''
        Validates that the value of k is greater than zero. THis returns
        the value of k if it's valid.

        Args:
            cls (type): The class of the model being validated.
            value (int): The value of k provided by the user.

        Returns:
            int: The value of k if it's valid.

        Raises:
            ValueError: If the value of k is not greater than zero.
        '''
        if value <= 0:
            raise ValueError("k must be greater than zero")
        return value

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        '''
        This is for the fit method, it takes in the observations and
        the ground truth and puts them in the parameters.

        Args:
            observations (np.ndarray): The observations
            ground_truth (np.ndarray): The ground truth

        Returns:
            None
        '''
        # This changes the values of parameters
        observations: np.ndarray = np.asarray(observations)
        ground_truth: np.ndarray = np.asarray(ground_truth)
        if self.k > len(ground_truth):
            raise ValueError("""k cannot be greater than the number
                             of training samples""")

        # Putting both the parameters in the dictionary and fitting them in
        self._parameters["observations"] = observations
        self._parameters["ground_truth"] = ground_truth

    def predict(self, observations: np.ndarray) -> np.ndarray:
        '''
        Predict the labels for the given observations.

        Args:
            observations (np.ndarray): Observations to predict

        Returns:
            np.ndarray: Predicted labels
        '''
        # This checks if the model has been fit or not
        if not self._parameters or "observations" not in self._parameters:
            raise ValueError("Model has not been fit")
        # This changes the values of parameters
        observations: np.ndarray = np.asarray(observations)
        predictions: np.ndarray = [
            self._predict_single(x) for x in observations
        ]
        return np.array(predictions)

    def _predict_single(self, observation: np.ndarray) -> int:
        '''
        This is for the predict method, it takes in a single observation
        and returns the predicted label for this observation
        args:
            observation (np.ndarray): The observation
        returns:
            int: The predicted label
        '''
        # This gets the distances between the observation
        distances: np.ndarray = np.linalg.norm(
            self._parameters["observations"] - observation, axis=1
        )
        k_indices: np.ndarray = np.argsort(distances)[:self.k]
        # This gets the labels of the k nearest
        # neighbors and stores them in a list
        k_nearest_labels: np.ndarray = (
            self._parameters["ground_truth"][k_indices]
        )
        most_common: Counter = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]