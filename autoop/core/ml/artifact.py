import os
import pickle
from typing import List, Dict, Any, Optional
import base64

class Artifact:
    """
    Artifact class for handling data storage and retrieval using pickle.
    """

    def __init__(
        self,
        name: str,
        asset_path: str,
        data: Any,
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
            data (Any): Data associated with the artifact.
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
        Saves the Artifact instance to a pickle file.

        Args:
            directory (str): The directory where the file will be saved.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f'{self.id}.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def read(cls, id: str, directory: str = 'artifacts') -> 'Artifact':
        """
        Reads the Artifact from a pickle file and recreates the Artifact instance.

        Args:
            id (str): The unique identifier of the artifact.
            directory (str): The directory where the file is located.

        Returns:
            Artifact: An instance of the Artifact class.
        """
        file_path = os.path.join(directory, f'{id}.pkl')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found for id {id}")
        with open(file_path, 'rb') as f:
            artifact = pickle.load(f)
        return artifact
