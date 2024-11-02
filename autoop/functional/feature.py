# autoop/functional/feature.py

from typing import List
from autoop.core.ml.feature import Feature

def detect_feature_types(dataset: 'Dataset') -> List[Feature]:
    """
    This is also enforced by the tests to be a function. Cannot
    do anything about it.
    
    """
    features = []
    data = dataset.to_dataframe()
    for column in data.columns:
        if data[column].dtype == 'object' or data[column].dtype.name == 'category':
            feature_type = 'categorical'
        else:
            feature_type = 'numerical'
        features.append(Feature(name=column, type=feature_type))
    return features
