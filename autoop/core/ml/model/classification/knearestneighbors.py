from .. import Model
import numpy as np
from collections import Counter

class KNearestNeighbors(Model):
    '''
    K-Nearest Neighbors classifier.
    '''
    type = "classification"

    def __init__(self, k=3, distance_metric='euclidean', weights='uniform', **kwargs) -> None:
        '''
        Initialize the KNN model with hyperparameters.

        Args:
            k (int): Number of neighbors to use.
            distance_metric (str): Distance metric to use ('euclidean', 'manhattan').
            weights (str): Weight function ('uniform', 'distance').
        '''
        super().__init__(**kwargs)
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        '''
        Fit the model with training data.

        Args:
            observations (np.ndarray): Training data features.
            ground_truth (np.ndarray): Training data labels.
        '''
        observations = np.asarray(observations)
        ground_truth = np.asarray(ground_truth)
        if self.k > len(ground_truth):
            raise ValueError("k cannot be greater than the number of training samples")
        if self.k <= 0:
            raise ValueError("k must be greater than zero")
        self.parameters["observations"] = observations
        self.parameters["ground_truth"] = ground_truth

    def predict(self, observations: np.ndarray) -> np.ndarray:
        '''
        Predict the labels for the given observations.

        Args:
            observations (np.ndarray): Observations to predict.

        Returns:
            np.ndarray: Predicted labels.
        '''
        if not self.parameters or "observations" not in self.parameters:
            raise ValueError("Model has not been fitted yet.")
        observations = np.asarray(observations)
        predictions = [self._predict_single(x) for x in observations]
        return np.array(predictions)

    def _compute_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        '''
        Compute the distance between two samples based on the distance metric.

        Args:
            x1 (np.ndarray): First sample.
            x2 (np.ndarray): Second sample.

        Returns:
            float: Computed distance.
        '''
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(x1 - x2)
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def _predict_single(self, observation: np.ndarray) -> int:
        '''
        Predict the label for a single observation.

        Args:
            observation (np.ndarray): The observation.

        Returns:
            int: The predicted label.
        '''
        distances = np.array([
            self._compute_distance(observation, train_obs)
            for train_obs in self.parameters["observations"]
        ])
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.parameters["ground_truth"][k_indices]
        if self.weights == 'uniform':
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]
        elif self.weights == 'distance':
            k_distances = distances[k_indices]
            k_distances = np.where(k_distances == 0, 1e-5, k_distances)  # Avoid division by zero
            weights = 1 / k_distances
            label_weights = {}
            for label, weight in zip(k_nearest_labels, weights):
                label_weights[label] = label_weights.get(label, 0) + weight
            return max(label_weights.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError(f"Unsupported weights: {self.weights}")
