import json
from typing import Dict, Tuple, List, Union
import os
from autoop.core.storage import Storage

class Database():

    def __init__(self, storage: Storage) -> None:
        self._storage = storage
        self._data = {}
        self._load()

    def set(self, collection: str, id: str, entry: dict) -> dict:
        """Set a key in the database
        Args:
            collection (str): The collection to store the data in
            id (str): The id of the data
            entry (dict): The data to store
        Returns:
            dict: The data that was stored
        """
        assert isinstance(entry, dict), "Data must be a dictionary"
        assert isinstance(collection, str), "Collection must be a string"
        assert isinstance(id, str), "ID must be a string"
        if not self._data.get(collection, None):
            self._data[collection] = {}
        self._data[collection][id] = entry
        self._persist()
        return entry

    def get(self, collection: str, id: str) -> Union[dict, None]:
        """Get a key from the database
        Args:
            collection (str): The collection to get the data from
            id (str): The id of the data
        Returns:
            Union[dict, None]: The data that was stored, or None if it doesn't exist
        """
        if not self._data.get(collection, None):
            return None
        return self._data[collection].get(id, None)

    def delete(self, collection: str, id: str):
        """
        Delete a key from the database
        Args:
            collection (str): The collection to delete the data from
            id (str): The id of the data
        Returns:
            None
        """
        if not self._data.get(collection, None):
            return
        if self._data[collection].get(id, None):
            del self._data[collection][id]
        self._persist()

    def list(self, collection: str) -> List[Tuple[str, dict]]:
        """Lists all data in a collection
        Args:
            collection (str): The collection to list the data from
        Returns:
            List[Tuple[str, dict]]: A list of tuples containing
            the id and data for each item in the collection
        """
        if not self._data.get(collection, None):
            return []
        return [(id, data) for id, data in self._data[collection].items()]

    def refresh(self) -> None:
        """Refresh the database by loading the data from storage"""
        self._load()

    def _persist(self) -> None:
        """Persist the data to storage"""
        for collection, data in self._data.items():
            if not data:
                continue
            for id, item in data.items():
                # Save each item using an OS-independent path
                path = os.path.join(collection, id)
                self._storage.save(json.dumps(item).encode(), path)

        # For things that were deleted, remove them from storage
        keys = self._storage.list("")
        for key in keys:
            # Normalize and split the path using OS-specific separators
            normalized_key = os.path.normpath(key)
            parts = normalized_key.split(os.sep)
            if len(parts) >= 2:
                collection, id = parts[-2], parts[-1]
                # Check if the data exists in memory, and if not, delete it
                if not self._data.get(collection, {}).get(id):
                    self._storage.delete(key)

    def _load(self) -> None:
        """Load the data from storage."""
        self._data = {}
        for key in self._storage.list(""):
            # Normalize the path to be platform-independent
            normalized_key = os.path.normpath(key)
            # Split the path to get the collection and id
            parts = normalized_key.split(os.sep)
            if len(parts) >= 2:
                collection, id = parts[-2], parts[-1]
                data = self._storage.load(key)
                # Ensure the collection exists in the dictionary
                if collection not in self._data:
                    self._data[collection] = {}
                self._data[collection][id] = json.loads(data.decode())
