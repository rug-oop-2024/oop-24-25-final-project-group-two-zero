from .. import Model
import numpy as np

class ConvlutedNeuralNetwork(Model):
    """
    Neural Network Classification model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv_filter = np.random.randn(3, 3)
        self.fc_weights = np.random.randn(169, 26)
        self.fc_bias = np.random.randn(26)
    
    def flatten_observations(self, observations: np.ndarray) -> np.ndarray:
        """
        This flattens the observations of the input data.

        Args:
            observations (np.ndarray): Observations (n_samples, n_features)

        Returns:
            np.ndarray: Flattened observations
        """
        return np.flatten(observations)
    
    # This is the maths of the Convluted Neural Network
    def relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax function for output layer."""
        exp_values = np.exp(x - np.max(x))
        return exp_values / np.sum(exp_values)
    
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        pass

        

    def predict(self, observations: np.ndarray) -> np.ndarray:
        pass
