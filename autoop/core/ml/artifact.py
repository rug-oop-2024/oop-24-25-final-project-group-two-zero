import os
import json
import h5py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from typing_extensions import Literal


class Artifact:
    """
    Artifact class for handling data storage and retrieval in HDF5 format.
    """

    def __init__(
        self,
        name: str,
        asset_path: str,
        data: bytes,
        version: str,
        type: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None
    ) -> None:
        """
        Initialize an Artifact instance.

        Args:
            name (str): Name of the artifact.
            asset_path (str): Path where artifact is stored.
            data (bytes): Data associated with the artifact.
            version (str): Version of the artifact.
            type (str): Type of artifact.
            tags (List[str], optional): Tags for categorizing the artifact.
            metadata (Dict[str, Any], optional): Additional metadata.
            id (str, optional): Unique identifier.
        """
        self.name = name
        self.asset_path = asset_path
        self.data = data
        self.version = version
        self.type = type
        self.tags = tags if tags is not None else []
        self.metadata = metadata if metadata is not None else {}
        self.id = id or os.urandom(16).hex()

    def save(self, directory: str = 'artifacts') -> None:
        """
        Saves the Artifact instance to an HDF5 file.

        Args:
            directory (str): The directory where the file will be saved.

        Not too sure this is correct
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f'{self.id}.h5')
        with h5py.File(file_path, 'w') as h5f:
            h5f.create_dataset('data', data=self.data)
            h5f.attrs['name'] = self.name
            h5f.attrs['asset_path'] = self.asset_path
            h5f.attrs['version'] = self.version
            h5f.attrs['type'] = self.type
            h5f.attrs['tags'] = json.dumps(self.tags)
            h5f.attrs['id'] = self.id
            h5f.attrs['metadata'] = json.dumps(self.metadata)

    @classmethod
    def read(cls, id: str, directory: str = 'artifacts') -> 'Artifact':
        """
        Reads the Artifact from an HDF5 file and recreates the Artifact instance.

        Args:
            id (str): The unique identifier of the artifact.
            directory (str): The directory where the file is located.

        Returns:
            Artifact: An instance of the Artifact class.
        """
        file_path = os.path.join(directory, f'{id}.h5')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found for id {id}")
        with h5py.File(file_path, 'r') as h5f:
            data = h5f['data'][()]
            name = h5f.attrs['name']
            asset_path = h5f.attrs['asset_path']
            version = h5f.attrs['version']
            type_ = h5f.attrs['type']
            tags = json.loads(h5f.attrs['tags'])
            id_ = h5f.attrs['id']
            metadata = json.loads(h5f.attrs['metadata'])
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
