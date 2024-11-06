from .model import Model
# Regression
from autoop.core.ml.model.regression import MultipleLinearRegression
from autoop.core.ml.model.regression import RidgeRegression
from autoop.core.ml.model.regression import SupportVectorRegression
# Classification
from autoop.core.ml.model.classification import TreeClassification
from autoop.core.ml.model.classification import KNearestNeighbors
from autoop.core.ml.model.classification import StochasticGradient
from autoop.core.ml.model.classification import TextClassificationModel


__all__ = [
    "Model",
    "MultipleLinearRegression",
    "RidgeRegression",
    "SupportVectorRegression",
    "TreeClassification",
    "KNearestNeighbors",
    "StochasticGradient",
    "TextClassificationModel"
]
