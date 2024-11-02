from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature

# Example implementation of detect_feature_types
# autoop/functional/feature.py
from typing import List
from autoop.core.ml.feature import Feature

class detect_feature_types:
    def __call__(dataset) -> List[Feature]:
        features = []
        data = dataset.to_dataframe()  # Convert bytes to DataFrame
        for column in data.columns:
            if data[column].dtype == 'object':
                feature_type = 'categorical'
            else:
                feature_type = 'numerical'
            features.append(Feature(name=column, type=feature_type))
        return features


