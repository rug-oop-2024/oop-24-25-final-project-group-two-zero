from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset

class Feature(BaseModel):
    name : str
    type : Literal["categorical", "numerical"]

    def __str__(self):
        return f"Feature(name={self.name}, type={self.type})"

    def __repr__(self):
        return self.__str__()