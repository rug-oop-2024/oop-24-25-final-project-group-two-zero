from .model import Model
# Regression
from autoop.core.ml.model.regression import LinearRegressionModel
from autoop.core.ml.model.regression import RidgeRegression
from autoop.core.ml.model.regression import LinearRegression
# Classification
from autoop.core.ml.model.classification import TreeClassification
from autoop.core.ml.model.classification import KNearestNeighbors
from autoop.core.ml.model.classification import StoasticGradient


__all__ = [
    "Model",
    "LinearRegressionModel",
    "RidgeRegression",
    "LinearRegression",
    "TreeClassification",
    "KNearestNeighbors",
    "StoasticGradient"
]
