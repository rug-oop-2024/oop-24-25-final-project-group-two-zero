# autoop/functional/feature.py

from typing import List
from autoop.core.ml.feature import Feature

class detect_feature_types:
    def __call__(self, dataset) -> List[Feature]:
        features = []
        data = dataset.to_dataframe()  # Convert bytes to DataFrame
        for column in data.columns:
            if data[column].dtype == 'object' or data[column].dtype.name == 'category':
                feature_type = 'categorical'
            else:
                feature_type = 'continuous'
            features.append(Feature(name=column, type=feature_type))
        return features
