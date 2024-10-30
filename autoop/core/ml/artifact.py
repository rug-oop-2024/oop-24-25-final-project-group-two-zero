# artifact.py
from pydantic import BaseModel, Field
import base64
import os
from typing_extensions import Literal
from typing import (List, Dict, Any, Optional, Union)
from abc import ABC, abstractmethod
from autoop.core.storage import Storage
import json
import h5py

# Do this later
class Artifact():
    """
    Treat this as a storage for all the information required
    to represent something. If something later on happens, change it again.
    Do it until you understand what the Artifact class is, and what it
    does.

    Make a decorator for the class name, do the same for the parameters, 
    and ensure that you can call the class inside of it. This can
    most likely be done by some static method or something of the like.
    use h5py

    """
    def __init__(
        self,
        name: str = None,
        asset_path = None,
        data = None,
        version = None,
        type = None,
        tags = None,
        metadata = None,
        id = None
        ):
        if id is None:
            id = os.urandom(16).hex()
        self.name: str = name
        self.asset_path: str = asset_path
        self.data: bytes = data
        self.version: str = version
        self.type: str = type
        self.tags: List[str] = tags
        self.metadata: Dict[str, Any] = metadata
        self.id: str = id



    def save(self, directory: str = 'artifacts'):
        """
        Saves the Artifact instance to an HDF5 file.

        Parameters:
            directory (str): The directory where the file will be saved.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f'{self.id}.h5')
        with h5py.File(file_path, 'w') as h5f:
            # Store binary data
            h5f.create_dataset('data', data=self.data)
            # Store attributes
            h5f.attrs['name'] = self.name
            h5f.attrs['asset_path'] = self.asset_path
            h5f.attrs['version'] = self.version
            h5f.attrs['type'] = self.type
            h5f.attrs['tags'] = json.dumps(self.tags)  # Convert list to JSON string
            h5f.attrs['id'] = self.id
            # Store metadata as JSON string
            h5f.attrs['metadata'] = json.dumps(self.metadata)

    @classmethod
    def read(cls, id: str, directory: str = 'artifacts'):
        """
        Reads the Artifact from an HDF5 file and recreates the Artifact instance.

        Parameters:
            id (str): The unique identifier of the artifact.
            directory (str): The directory where the file is located.

        Returns:
            Artifact: An instance of the Artifact class.
        """
        file_path = os.path.join(directory, f'{id}.h5')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found for id {id}")
        with h5py.File(file_path, 'r') as h5f:
            # Read data
            data = h5f['data'][()]
            # Read attributes
            name = h5f.attrs['name']
            asset_path = h5f.attrs['asset_path']
            version = h5f.attrs['version']
            type_ = h5f.attrs['type']
            tags = json.loads(h5f.attrs['tags'])  # Convert JSON string back to list
            id_ = h5f.attrs['id']
            # Read metadata
            metadata = json.loads(h5f.attrs['metadata'])  # Convert JSON string back to dict
            # Create Artifact instance
            return cls(
                name=name,
                asset_path=asset_path,
                data=data,
                version=version,
                type=type_,
                tags=tags,
                metadata=metadata,
                id=id_
            )
    
    """
    self innitialize the variable using something like this

    @staticmethod
    def get_instance():
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"), 
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    """
