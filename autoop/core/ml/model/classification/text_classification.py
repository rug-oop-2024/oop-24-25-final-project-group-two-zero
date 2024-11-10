from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from ..model import Model
import numpy as np
from typing import Any, List


class TextClassificationModel(Model):
    """Text classification model using TF-IDF and Logistic Regression."""

    _type: str = "classification"
    _available_hyperparameters: dict = {
        "max_features": 10000,
        "ngram_range": (1, 1),
        "C": 1.0,
        "penalty": ["l2", "l1", "elasticnet"],
        "solver": ["liblinear", "saga"],
        "epochs": 10,
    }

    supported_feature_types: List[str] = ["text"]
    supported_target_types: List[str] = ["categorical"]

    def __init__(self: "TextClassificationModel", **hyperparameters: Any) -> None:
        """
        Initialize the TextClassificationModel with hyperparameters.

        Args:
            **hyperparameters: Hyperparameters for the model.
        """
        super().__init__(**hyperparameters)
        self._hyperparameters = {
            **self._available_hyperparameters,
            **self._hyperparameters,
        }
        self._vectorizer = None
        self._model = None
        self._build_model()

    def _build_model(self: "TextClassificationModel") -> None:
        """
        Build the TextClassificationModel based
        on the given hyperparameters.
        """
        max_features = self._hyperparameters["max_features"]
        ngram_range = self._hyperparameters["ngram_range"]
        C = self._hyperparameters["C"]
        penalty = self._hyperparameters["penalty"]
        solver = self._hyperparameters["solver"]

        self._vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range
        )
        self._model = LogisticRegression(
            C=C, penalty=penalty, solver=solver, max_iter=1000
        )

    def fit(
        self: "TextClassificationModel",
        observations: List[str],
        ground_truth: np.ndarray,
    ) -> None:
        """
        Fits the model to the given data.

        Args:
            observations (List[str]):
                The input text data to fit the model to.
            ground_truth (np.ndarray):
                The target values to fit the model to.
        """
        X: np.ndarray = self._vectorizer.fit_transform(observations)
        self._model.fit(X, ground_truth)
        self._parameters: dict = {
            "coef_": self._model.coef_,
            "intercept_": self._model.intercept_,
            "classes_": self._model.classes_,
        }

    def predict(self: "TextClassificationModel", observations: List[str]) -> np.ndarray:
        """
        Predicts the labels for the given observations.

        Args:
            observations (List[str]): The input text data to predict.

        Returns:
            np.ndarray: The predicted labels.
        """
        X: np.ndarray = self._vectorizer.transform(observations)
        return self._model.predict(X)
