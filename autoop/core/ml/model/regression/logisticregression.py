from .. import Model
import numpy as np

class LogisticRegression(Model):
    def __init__(self, num_iterations: int = 1000, learning_rate: float = 0.01) -> None:
        super().__init__()
        self.theta: np.ndarray | None = None
        self._numiterations: int = num_iterations
        self._learning_rate: float = learning_rate

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        This calculates the sigmoin function for z

        Args:
            z (np.ndarray): z value

        Returns:
            np.ndarray: sigmoid(z)
        """
        return 1 / (1 + np.exp(-z))
    
    def cost(self, observations: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Cost function for logistic regression

        Args:
            observations (np.ndarray): Observations (n_samples, n_features)
            ground_truth (np.ndarray): Ground truth (n_samples, )

        Returns:
            float: Cost
        """
        m: int = ground_truth.shape[0]
        logging_values: np.ndarray = np.log(1-observations)
        normal_log = np.log(observations)
        cost = (-1/m) * np.sum(ground_truth * normal_log + (1-ground_truth) * logging_values)
        return cost
    
    def gradient_decent(
            self,
            observations: np.ndarray,
            predictions: np.ndarray,
            ground_truth: np.ndarray
            ) -> np.ndarray:
        """
        Gradient descent for logistic regression

        Args:
            observations (np.ndarray): Observations (n_samples, n_features)
            ground_truth (np.ndarray): Ground truth (n_samples, )

        Returns:
            np.ndarray: Gradient
        """
        m = ground_truth.shape[0]
        gradient = (1/m) * (observations.T @ (predictions - ground_truth))
        return gradient
    



    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the logistic regression model using the normal equation.

        Args:
            observations (np.ndarray): Observations (n_samples, n_features)
            ground_truth (np.ndarray): Ground truth (n_samples, )

        Returns:
            None

        Stores:
            self._parameters['parameters'] (np.ndarray): Coefficient vector
            including intercept term
        """
        self.theta = np.zeros(observations.shape[1])

        for i in range(self._numiterations):
            z = np.dot(observations, self.theta)
            predictions = self.sigmoid(z)
            gradient = self.gradient_decent(predictions,predictions, ground_truth)
            self.theta = self.theta - (self._learning_rate * gradient)
            if i % 100 == 0:
                cost = self.cost(observations, ground_truth)
                print("Cost after iteration %i: %f" %(i, cost))
    
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the given observations.

        Args:
            observations (np.ndarray): Observations to predict

        Returns:
            np.ndarray: Predicted labels
        """
        # This checks if the model has been fit or not
        if not self._parameters or "observations" not in self._parameters:
            raise ValueError("Model has not been fit")
        z = np.dot(observations, self.theta)
        probability = self.sigmoid(z)
        return [1 if x >= 0.5 else 0 for x in probability]

        



