from pydantic import BaseModel
from typing import Literal


class Feature(BaseModel):
    name: str
    type: Literal['categorical', 'numerical','image', 'text', 'audio', 'video']
    is_target: bool = False

    def __str__(self) -> str:
        """Returns a string representation of the feature in the format 'Feature(name=foo, type=bar, is_target=baz)'. This is used for debugging purposes."""
        return f"Feature(name={self.name}, type={self.type}, is_target={self.is_target})"

    def __repr__(self) -> str:
        return self.__str__()
