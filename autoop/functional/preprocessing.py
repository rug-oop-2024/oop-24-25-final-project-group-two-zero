from typing import List, Tuple
from autoop.core.ml.feature import Feature
from autoop.core.ml.dataset import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocess_features(features: List[Feature], dataset: Dataset) -> List[Tuple[str, np.ndarray, dict]]:
    """
    This is forced to be a function so that it can be called in the pipeline,
    and I cannot change pipeline.
    
    
    """
    results = []
    raw = dataset.to_dataframe()  # Use to_dataframe() instead of read()
    for feature in features:
        if feature.type == "categorical":
            encoder = OneHotEncoder()
            data = encoder.fit_transform(raw[feature.name].values.reshape(-1, 1)).toarray()
            artifact = {"type": "OneHotEncoder", "encoder": encoder}  # Store the actual encoder
            results.append((feature.name, data, artifact))
        elif feature.type == "numerical":
            scaler = StandardScaler()
            data = scaler.fit_transform(raw[feature.name].values.reshape(-1, 1))
            artifact = {"type": "StandardScaler", "scaler": scaler}  # Store the actual scaler
            results.append((feature.name, data, artifact))
        else:
            raise ValueError(f"Unknown feature type: {feature.type}")
    results = list(sorted(results, key=lambda x: x[0]))
    return results
