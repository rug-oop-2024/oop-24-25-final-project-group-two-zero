# autoop/functional/feature.py
from typing import List, Literal
from autoop.core.ml.feature import Feature
from pydantic import BaseModel

class Feature(BaseModel):
    name: str
    type: Literal['categorical', 'numerical']
    is_target: bool = False

    def __str__(self) -> str:
        return f"Feature(name={self.name}, type={self.type}, is_target={self.is_target})"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
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
