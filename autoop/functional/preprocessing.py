from typing import List, Tuple, Optional
from autoop.core.ml.feature import Feature
from autoop.core.ml.dataset import Dataset
import numpy as np
from sklearn.preprocessing import (
                OneHotEncoder,
                StandardScaler,
                LabelEncoder
                )


def preprocess_features(
    features: List[Feature], dataset: Dataset
) -> Optional[List[Tuple[str, np.ndarray, dict]]]:
    """
    Preprocesses features in a dataset based on feature type.

    Args:
        features (List[Feature]): List of features to preprocess.
        dataset (Dataset): Dataset to preprocess.

    Returns:
        Optional[List[Tuple[str, np.ndarray, dict]]]:
            A list of tuples containing
        the feature name, transformed data,
        and the preprocessing artifact for
        each feature, or None if a ValueError is raised.
    """
    results = []
    df = dataset.to_dataframe()

    for feature in features:
        if feature.type == "categorical":
            if getattr(feature, "is_target", False):
                # Use LabelEncoder for the target feature
                encoder = LabelEncoder()
                data = encoder.fit_transform(df[feature.name])
                artifact = {"type": "LabelEncoder", "encoder": encoder}
            else:
                # Use OneHotEncoder for input features
                encoder = OneHotEncoder(sparse_output=False)
                data = encoder.fit_transform(df[[feature.name]])
                artifact = {"type": "OneHotEncoder", "encoder": encoder}
        elif feature.type == "numerical":
            # For numerical features
            scaler = StandardScaler()
            data = scaler.fit_transform(df[[feature.name]])
            artifact = {"type": "StandardScaler", "scaler": scaler}
        else:
            raise ValueError(f"Unknown feature type: {feature.type}")

        results.append((feature.name, data, artifact))

    return results
