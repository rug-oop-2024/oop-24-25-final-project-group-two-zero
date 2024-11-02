import unittest
import numpy as np
from .knearestneighbors import KNearestNeighbors

class TestKNearestNeighbors(unittest.TestCase):

    def setUp(self) -> None:
        """
        Setup the test by initializing the KNearestNeighbors model with k=3 and preparing the test data.
        The test data consists of 5 observations with 2 features each, and their corresponding ground truth labels.
        The observations are:
            [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        The ground truth labels are:
            [0, 1, 0, 1, 0]
        """
        
        self.knn = KNearestNeighbors(k=3)
        self.observations = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        self.ground_truth = np.array([0, 1, 0, 1, 0])

    def test_fit(self):
        """
        Test the fit method of the KNearestNeighbors model.

        This test ensures that after fitting the model with the provided observations
        and ground truth labels, the parameters 'observations' and 'ground_truth' 
        stored in the model are equal to the input data.
        """
        self.knn.fit(self.observations, self.ground_truth)
        np.testing.assert_array_equal(self.knn.parameters["observations"], self.observations)
        np.testing.assert_array_equal(self.knn.parameters["ground_truth"], self.ground_truth)

    def test_predict(self):
        """
        Test the predict method of the KNearestNeighbors model.

        This test ensures that the predict method returns an array of
        predicted labels for the given observations. The test first fits
        the model with the provided observations and ground truth labels.
        Then it uses the predict method to get the predicted labels for
        two test observations. The test asserts that the shape of the
        returned array is equal to the number of test observations, and
        that the predicted labels are either 0 or 1.
        """
        self.knn.fit(self.observations, self.ground_truth)
        predictions = self.knn.predict(np.array([[2, 3], [3, 4]]))
        self.assertEqual(predictions.shape[0], 2)
        self.assertIn(predictions[0], [0, 1])
        self.assertIn(predictions[1], [0, 1])

    def test_invalid_k(self):
        """
        Test that the KNearestNeighbors model raises a ValueError when the k parameter is
        set to an invalid value.

        The test ensures that the model raises a ValueError when k is set to 0, and
        when k is set to a value greater than the number of observations in the
        training data. In this case, the test data contains 5 observations, so
        the model should raise a ValueError when k is set to 6.
        """
        with self.assertRaises(ValueError):
            KNearestNeighbors(k=0).fit(self.observations, self.ground_truth)
        with self.assertRaises(ValueError):
            KNearestNeighbors(k=6).fit(self.observations, self.ground_truth)

    def test_unsupported_distance_metric(self):
        """
        Test that the KNearestNeighbors model raises a ValueError when an unsupported
        distance metric is provided.

        This test initializes the model with an unsupported distance metric and fits
        it with observations and ground truth labels. It asserts that a ValueError is
        raised when attempting to predict with this configuration.
        """
        knn = KNearestNeighbors(k=3, distance_metric='unsupported')
        knn.fit(self.observations, self.ground_truth)
        with self.assertRaises(ValueError):
            knn.predict(np.array([[2, 3]]))

    def test_unsupported_weights(self):
        """
        Test that the KNearestNeighbors model raises a ValueError when an unsupported
        weights parameter is provided.

        This test initializes the model with an unsupported weights parameter and fits
        it with observations and ground truth labels. It asserts that a ValueError is
        raised when attempting to predict with this configuration.
        """
        
        knn = KNearestNeighbors(k=3, weights='unsupported')
        knn.fit(self.observations, self.ground_truth)
        with self.assertRaises(ValueError):
            knn.predict(np.array([[2, 3]]))

if __name__ == '__main__':
    unittest.main()