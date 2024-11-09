from pydantic import BaseModel
from typing import Literal


class Feature(BaseModel):
    name: str
    type: Literal["categorical",
                "numerical", "image", "text",
                "audio", "video"]
    is_target: bool = False

    def __str__(self) -> str:
        """Returns a string representation of
        the feature in the format 'name'.
        This is used for debugging purposes."""
        return f"{self.name}"

    def __repr__(self) -> str:
        """Returns a string representation of the feature,
        useful for debugging purposes.
        This is the same as str(self)."""
        return self.__str__()
