import unittest
import numpy as np
from .. import Model

class DummyModel(Model):
    def fit(self, observations: np.ndarray, groundtruth: np.ndarray) -> None:
        self.parameters['mean'] = np.mean(observations, axis=0)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return np.full(observations.shape[0], self.parameters.get('mean', 0))

class TestModel(unittest.TestCase):

    def setUp(self) -> None:
        self.model = DummyModel()
        self.observations = np.array([[1, 2], [3, 4], [5, 6]])
        self.groundtruth = np.array([1, 0, 1])

    def test_fit(self):
        self.model.fit(self.observations, self.groundtruth)
        self.assertIn('mean', self.model.parameters)
        np.testing.assert_array_equal(self.model.parameters['mean'], np.array([3, 4]))

    def test_predict(self):
        self.model.fit(self.observations, self.groundtruth)
        predictions = self.model.predict(self.observations)
        self.assertEqual(predictions.shape[0], self.observations.shape[0])
        np.testing.assert_array_equal(predictions, np.array([3, 3, 3]))

if __name__ == '__main__':
    unittest.main()