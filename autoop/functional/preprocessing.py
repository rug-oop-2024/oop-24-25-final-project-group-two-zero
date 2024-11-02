from typing import List, Tuple
from autoop.core.ml.feature import Feature
from autoop.core.ml.dataset import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

def preprocess_features(features: List[Feature], dataset: Dataset) -> List[Tuple[str, np.ndarray, dict]]|None:
    """
    This is enforced by the tests to be a function. Cannot
    do anything about it.

    Args:
        features (List[Feature]): List of features
        dataset (Dataset): Dataset to preprocess

    returns:
        List[Tuple[str, np.ndarray, dict]]
        None if ValueError is raised
    """
    results = []
    df = dataset.to_dataframe()
    for feature in features:
        if feature.type == "categorical":
            if getattr(feature, 'is_target', False):
                # Use LabelEncoder for the target feature
                encoder = LabelEncoder()
                data = encoder.fit_transform(df[feature.name])
                artifact = {
                    "type": "LabelEncoder",
                    "encoder": encoder
                }
            else:
                # Use OneHotEncoder for input features
                encoder = OneHotEncoder(sparse_output=False)
                data = encoder.fit_transform(df[[feature.name]])
                artifact = {
                    "type": "OneHotEncoder",
                    "encoder": encoder
                }
        elif feature.type == "numerical":
            # For numerical features
            scaler = StandardScaler()
            data = scaler.fit_transform(df[[feature.name]])
            artifact = {
                "type": "StandardScaler",
                "scaler": scaler
            }
        else:
            raise ValueError(f"Unknown feature type: {feature.type}")
        results.append((feature.name, data, artifact))
    return results
