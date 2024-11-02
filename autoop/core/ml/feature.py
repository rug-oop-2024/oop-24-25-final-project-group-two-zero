# feature.py

from pydantic import BaseModel
from typing import Literal

class Feature(BaseModel):
    name: str
    type: Literal['categorical', 'numerical']
    is_target: bool = False

    def __str__(self) -> str:
        return f"Feature(name={self.name}, type={self.type}, is_target={self.is_target})"

    def __repr__(self) -> str:
        return self.__str__()
