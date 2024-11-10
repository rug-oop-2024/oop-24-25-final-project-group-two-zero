import unittest
import numpy as np
from .. import Model


class DummyModel(Model):
    def fit(self, observations: np.ndarray, groundtruth: np.ndarray) -> None:
        """
        Computes the mean of the observations and stores it in the parameters.

        Args:
            observations (np.ndarray): The input data to compute the mean from.
            groundtruth (np.ndarray): The target values (not used in this implementation).
        """
        self.parameters["mean"] = np.mean(observations, axis=0)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Makes predictions based on the mean of the observations.

        Args:
            observations (np.ndarray): Features for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        return np.full(observations.shape[0], self.parameters.get("mean", 0))


class TestModel(unittest.TestCase):

    def setUp(self) -> None:
        """
        Setup the test by initializing a DummyModel and two arrays of features and targets.

        The features are:
            [[1, 2], [3, 4], [5, 6]]
        The targets are:
            [1, 0, 1]
        """
        self.model = DummyModel()
        self.observations = np.array([[1, 2], [3, 4], [5, 6]])
        self.groundtruth = np.array([1, 0, 1])

    def test_fit(self):
        """
        Tests that the fit method stores the mean of the observations in the
        parameters of the model. The mean should be [3, 4] for the test data.
        """
        self.model.fit(self.observations, self.groundtruth)
        self.assertIn("mean", self.model.parameters)
        np.testing.assert_array_equal(self.model.parameters["mean"], np.array([3, 4]))

    def test_predict(self):
        """
        Tests that the predict method makes predictions based on the mean of the observations.

        The test first fits the model with the provided features and targets. Then it uses the predict
        method to get the predicted values for the same features. The test asserts that the predicted
        values are equal to the mean of the observations, which is [3, 4] for the test data.

        The test also asserts that the shape of the predicted values is equal to the number of observations.
        """
        self.model.fit(self.observations, self.groundtruth)
        predictions = self.model.predict(self.observations)
        self.assertEqual(predictions.shape[0], self.observations.shape[0])
        np.testing.assert_array_equal(predictions, np.array([3, 3, 3]))


if __name__ == "__main__":
    unittest.main()
