import unittest
import numpy as np
from .LinearRegression import LinearRegressionModel

class TestLinearRegressionModel(unittest.TestCase):

    def setUp(self) -> None:
        """
        Setup the test by initializing the LinearRegressionModel with default values, and preparing the test data.
        The test data consists of 4 observations with 2 features each, and their corresponding ground truth labels.
        The observations are:
            [[1, 1], [2, 2], [3, 3], [4, 4]]
        The ground truth labels are:
            [2, 4, 6, 8]
        """
        self.model = LinearRegressionModel()
        self.observations = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        self.ground_truth = np.array([2, 4, 6, 8])

    def test_fit(self):
        """
        Test the fit method of the LinearRegressionModel.

        This test ensures that after fitting the model with the provided observations
        and ground truth labels, the parameters 'coef_' and 'intercept_' stored in the model
        are equal to the expected values.
        """
        self.model.fit(self.observations, self.ground_truth)
        self.assertIn('coef_', self.model.parameters)
        self.assertIn('intercept_', self.model.parameters)
        np.testing.assert_array_almost_equal(self.model.parameters['coef_'], np.array([2, 2]))
        self.assertAlmostEqual(self.model.parameters['intercept_'], 0)

    def test_predict(self):
        """
        Test the predict method of the LinearRegressionModel.

        This test ensures that after fitting the model with the provided observations
        and ground truth labels, the predict method returns the expected predictions
        for new observations.

        The test first fits the model with the provided data, and then uses the predict
        method to get the predicted labels for two new test observations. The test asserts
        that the predicted labels are equal to the expected values.
        """
        self.model.fit(self.observations, self.ground_truth)
        predictions = self.model.predict(np.array([[5, 5], [6, 6]]))
        np.testing.assert_array_almost_equal(predictions, np.array([10, 12]))

if __name__ == '__main__':
    unittest.main()