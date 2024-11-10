import unittest
import numpy as np
from autoop.core.ml.metric import (
    MeanSquaredError,
    MeanAbsoluteError,
    R2Score,
    Accuracy,
    Specificity,
    F1Score,
)


class TestMeanSquaredError(unittest.TestCase):

    def setUp(self) -> None:
        """
        Sets up the testing environment by initializing the MeanSquaredError metric.
        """
        self.metric = MeanSquaredError()

    def test_mse_perfect(self):
        """
        Tests that the mean squared error is 0.0 when the predictions are perfect.
        """
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 0.0)

    def test_mse_non_perfect(self):
        """
        Tests that the mean squared error is 0.25 when the predictions are not perfect.
        """
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([2, 2, 3, 4])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 0.25)


class TestMeanAbsoluteError(unittest.TestCase):

    def setUp(self) -> None:
        """
        Sets up the testing environment by initializing the MeanAbsoluteError metric.
        """
        self.metric = MeanAbsoluteError()

    def test_mae_perfect(self):
        """
        Tests that the mean absolute error is 0.0 when the predictions are perfect.
        """
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 0.0)

    def test_mae_non_perfect(self):
        """
        Tests that the mean absolute error is 0.25 when the predictions
        deviate from the true values.
        """
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([2, 2, 3, 4])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 0.25)


class TestR2Score(unittest.TestCase):

    def setUp(self) -> None:
        """
        Sets up the testing environment by initializing the R2Score metric.
        """
        self.metric = R2Score()

    def test_r2_perfect(self):
        """
        Tests that the R2 score is 1.0 when the predictions are perfect.
        """
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 1.0)

    def test_r2_non_perfect(self):
        """
        Tests that the R2 score is 0.9285714285714286 when the predictions deviate from the true values.
        """
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([2, 2, 3, 4])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertAlmostEqual(score, 0.9285714285714286)


class TestAccuracy(unittest.TestCase):

    def setUp(self) -> None:
        """
        Sets up the testing environment by initializing the Accuracy metric.
        """
        self.metric = Accuracy()

    def test_accuracy_perfect(self):
        """
        Tests that the accuracy is 1.0 when the predictions are perfect.
        """
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 1.0)

    def test_accuracy_non_perfect(self):
        """
        Tests that the accuracy is 0.75 when the predictions are not perfect.
        """
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 0])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 0.75)


class TestSpecificity(unittest.TestCase):

    def setUp(self) -> None:
        """
        Sets up the testing environment by initializing the Specificity metric.
        """
        self.metric = Specificity()

    def test_specificity_perfect(self):
        """
        Tests that the specificity is 1.0 when all true and predicted values are negative.
        """
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 1.0)

    def test_specificity_non_perfect(self):
        """
        Tests that the specificity is 0.75 when one of the predicted negative values is incorrect.
        """
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 1, 0, 0])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 0.75)


class TestF1Score(unittest.TestCase):

    def setUp(self) -> None:
        """
        Sets up the testing environment by initializing the F1Score metric.
        """
        self.metric = F1Score()

    def test_f1_score_perfect(self):
        """
        Tests that the F1 score is 1.0 when all true and predicted values match.
        """
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 1.0)

    def test_f1_score_zero(self):
        """
        Tests that the F1 score is 0.0 when there are no true positives.
        asserts all values to check if they are equal
        """
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 0.0)

    def test_f1_score_mixed(self):
        """
        Tests that the F1 score is 0.8 when there are 2 true positives and 2 false positives.
        """
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        score = self.metric.evaluate(y_pred, y_true)
        expected_score = 2 * (2 / 3 * 2 / 3) / (2 / 3 + 2 / 3)
        self.assertAlmostEqual(score, expected_score)

    def test_f1_score_no_positives(self):
        """
        Tests that the F1 score is 0.0 when there are no positive predictions or true positives.
        """
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
