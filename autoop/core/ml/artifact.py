from pydantic import BaseModel, Field
import base64
import os
from typing_extensions import Literal
# Do this later
class Artifact(BaseModel):
    relative_path: str = os.path.basename(__file__) # Done
    data
    type: str

    def read(self) -> bytes:
        return base64.b64decode(self.data)

    def save(self, data: bytes) -> str:
        # self.data = base64.b64encode(data).decode()
        # return self.data
        return base64.b64encode(data).decode()
    
