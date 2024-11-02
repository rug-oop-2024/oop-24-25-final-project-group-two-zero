from typing import List, Literal
from autoop.core.ml.feature import Feature
from pydantic import BaseModel


class Feature(BaseModel):
    name: str
    type: Literal['categorical', 'numerical']
    is_target: bool = False

    def __str__(self) -> str:
        """
        Returns a string representation of the Feature instance.

        The string representation includes the feature name, type, and a flag indicating
        whether it is the target feature.

        Returns:
            str: A string in the format 'Feature(name=foo, type=bar, is_target=baz)'.
        """
        return f"Feature(name={self.name}, type={self.type}, is_target={self.is_target})"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def detect_feature_types(dataset: 'Dataset') -> List['Feature']:
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
            if data[column].dtype == 'object' or data[column].dtype.name == 'category':
                feature_type = 'categorical'
            else:
                feature_type = 'numerical'
            features.append(Feature(name=column, type=feature_type))
        return features
