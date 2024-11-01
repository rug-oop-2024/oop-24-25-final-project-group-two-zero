from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    def __call__(self) -> List[Feature]:
        """Detects feature types for each feature in the dataset.
        Returns:
            List[Feature]: List of features with their types.
        """
        """
        Then I would need to put in the type of the feature, so 
        numbers or categorical
        
        """
        features = []
        dataset_to_use = dataset.read()
        dataset_to_use.to_csv(index=False).decode()
        columns = dataset_to_use.columns # This finds the column names in the dataset
        for column in columns:
            if dataset_to_use[column].dtype == object:
                features.append(Feature(name=column, type="categorical"))
            else:
                features.append(Feature(name=column, type="numerical"))
        return features
