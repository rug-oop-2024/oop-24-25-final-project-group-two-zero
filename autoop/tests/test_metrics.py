import unittest
import numpy as np
from autoop.core.ml.metric import (
    MeanSquaredError, MeanAbsoluteError, R2Score, Accuracy, Specificity, F1Score
)

class TestMeanSquaredError(unittest.TestCase):

    def setUp(self) -> None:
        self.metric = MeanSquaredError()

    def test_mse_perfect(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 0.0)

    def test_mse_non_perfect(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([2, 2, 3, 4])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 0.25)

class TestMeanAbsoluteError(unittest.TestCase):

    def setUp(self) -> None:
        self.metric = MeanAbsoluteError()

    def test_mae_perfect(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 0.0)

    def test_mae_non_perfect(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([2, 2, 3, 4])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 0.25)

class TestR2Score(unittest.TestCase):

    def setUp(self) -> None:
        self.metric = R2Score()

    def test_r2_perfect(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 1.0)

    def test_r2_non_perfect(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([2, 2, 3, 4])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertAlmostEqual(score, 0.9285714285714286)

class TestAccuracy(unittest.TestCase):

    def setUp(self) -> None:
        self.metric = Accuracy()

    def test_accuracy_perfect(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 1.0)

    def test_accuracy_non_perfect(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 0])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 0.75)

class TestSpecificity(unittest.TestCase):

    def setUp(self) -> None:
        self.metric = Specificity()

    def test_specificity_perfect(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 1.0)

    def test_specificity_non_perfect(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 1, 0, 0])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 0.75)

class TestF1Score(unittest.TestCase):

    def setUp(self) -> None:
        self.metric = F1Score()

    def test_f1_score_perfect(self):
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 1.0)

    def test_f1_score_zero(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 0.0)

    def test_f1_score_mixed(self):
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        score = self.metric.evaluate(y_pred, y_true)
        expected_score = 2 * (2/3 * 2/3) / (2/3 + 2/3)
        self.assertAlmostEqual(score, expected_score)

    def test_f1_score_no_positives(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        score = self.metric.evaluate(y_pred, y_true)
        self.assertEqual(score, 0.0)

if __name__ == "__main__":
    unittest.main()