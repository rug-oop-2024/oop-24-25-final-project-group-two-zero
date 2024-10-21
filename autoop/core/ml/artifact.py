from pydantic import BaseModel, Field
import base64
import os
from typing_extensions import Literal
from typing import (List, Dict, Any, Optional, Union)

# Do this later
class Artifact(BaseModel):
    """
    Treat this as a storage for all the information required
    to represent something. If something later on happens, change it again.
    Do it until you understand what the Artifact class is, and what it
    does.
    """

    relative_path: str = os.path.basename(__file__) # Done
    data = None
    type: str
    _parameters = {
        'data': None,
        'type': None,
        'relative_path': None


    }

    def read(self) -> bytes:
        """
        Reads the artifact and returns a bytes representation of it.
        """
        return base64.b64decode(self.data)

    def save(self, data: bytes) -> str:
        # self.data = base64.b64encode(data).decode()
        # return self.data
        self._parameters = {
            'data': base64.b64encode(data).decode()
        }
        
    
