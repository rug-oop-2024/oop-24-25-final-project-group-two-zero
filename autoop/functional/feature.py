from typing import List, Literal
from autoop.core.ml.feature import Feature
from pydantic import BaseModel


def detect_feature_types(dataset: "Dataset") -> List["Feature"]:
    """
    Detects the feature types in a dataset.

    Args:
        dataset (Dataset): The dataset to detect feature types in.

    Returns:
        List[Feature]: A list of Feature objects, each representing a column in the dataset.
    """
    features = []
    data = dataset.to_dataframe()
    for column in data.columns:
        if data[column].dtype == "object" or data[column].dtype.name == "category":
            feature_type = "categorical"
        else:
            feature_type = "numerical"
        features.append(Feature(name=column, type=feature_type))
    return features
