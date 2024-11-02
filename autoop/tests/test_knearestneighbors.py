import unittest
import numpy as np
from .knearestneighbors import KNearestNeighbors

class TestKNearestNeighbors(unittest.TestCase):

    def setUp(self) -> None:
        self.knn = KNearestNeighbors(k=3)
        self.observations = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        self.ground_truth = np.array([0, 1, 0, 1, 0])

    def test_fit(self):
        self.knn.fit(self.observations, self.ground_truth)
        np.testing.assert_array_equal(self.knn.parameters["observations"], self.observations)
        np.testing.assert_array_equal(self.knn.parameters["ground_truth"], self.ground_truth)

    def test_predict(self):
        self.knn.fit(self.observations, self.ground_truth)
        predictions = self.knn.predict(np.array([[2, 3], [3, 4]]))
        self.assertEqual(predictions.shape[0], 2)
        self.assertIn(predictions[0], [0, 1])
        self.assertIn(predictions[1], [0, 1])

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            KNearestNeighbors(k=0).fit(self.observations, self.ground_truth)
        with self.assertRaises(ValueError):
            KNearestNeighbors(k=6).fit(self.observations, self.ground_truth)

    def test_unsupported_distance_metric(self):
        knn = KNearestNeighbors(k=3, distance_metric='unsupported')
        knn.fit(self.observations, self.ground_truth)
        with self.assertRaises(ValueError):
            knn.predict(np.array([[2, 3]]))

    def test_unsupported_weights(self):
        knn = KNearestNeighbors(k=3, weights='unsupported')
        knn.fit(self.observations, self.ground_truth)
        with self.assertRaises(ValueError):
            knn.predict(np.array([[2, 3]]))

if __name__ == '__main__':
    unittest.main()