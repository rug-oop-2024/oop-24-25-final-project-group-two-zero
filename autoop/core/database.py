import json
from typing import Dict, Tuple, List, Union
import os
from autoop.core.storage import Storage


class Database:
    """
    Database class for persisting and retrieving data.

    This class is responsible for storing and retrieving data from a
    storage implementation.
    """

    def __init__(self: "Database", storage: Storage) -> None:
        """
        Initialize a Database instance.

        Args:
            storage (Storage): The storage
            instance for persisting and retrieving data.
        """
        self._storage = storage
        self._data: Dict[str, Dict[str, dict]] = {}
        self._load()

    def set(self: "Database", collection: str, id: str, entry: dict) -> dict:
        """
        Set a key in the database.

        Args:
            collection (str):
                The collection to store the data in.
            id (str): The id of the data.
            entry (dict): The data to store.

        Returns:
            dict: The data that was stored.
        """
        assert isinstance(entry, dict), "Data must be a dictionary"
        assert isinstance(collection, str), "Collection must be a string"
        assert isinstance(id, str), "ID must be a string"
        if collection not in self._data:
            self._data[collection] = {}
        self._data[collection][id] = entry
        self._persist()
        return entry

    def get(self: "Database", collection: str, id: str) -> Union[dict, None]:
        """
        Get a key from the database.

        Args:
            collection (str): The collection to get the data from.
            id (str): The id of the data.

        Returns:
            Union[dict, None]: The data that was stored, or None
                if it doesn't exist.
        """
        return self._data.get(collection, {}).get(id)

    def delete(self: "Database", collection: str, id: str) -> None:
        """
        Delete a key from the database.

        Args:
            collection (str): The collection to delete
                the data from.
            id (str): The id of the data.
        """
        if collection in self._data and id in self._data[collection]:
            del self._data[collection][id]
            self._persist()

    def list(self: "Database", collection: str) -> List[Tuple[str, dict]]:
        """
        List all data in a collection.

        Args:
            collection (str): The collection to list the data from.

        Returns:
            List[Tuple[str, dict]]: A list of tuples
                containing the id and data for each item
                in the collection.
        """
        if collection not in self._data:
            return []
        return list(self._data[collection].items())

    def refresh(self: "Database") -> None:
        """
        Refresh the database by reloading data from the storage.

        This method is useful if another process has
        modified the storage and
        you want to make sure the database is up to date.
        It will discard any
        unsaved changes you may have made to the database.
        """
        self._load()

    def _persist(self: "Database") -> None:
        """
        Persist the current state of the database to storage.

        This method iterates over all collections
        and their corresponding data,
        saving each item to storage using its
        collection and id as the path.
        After saving, it also removes any items from
        storage that are no longer
        present in the in-memory database.

        Returns:
            None
        """
        for collection, data in self._data.items():
            if not data:
                continue
            for id, item in data.items():
                path = os.path.join(collection, id)
                self._storage.save(json.dumps(item).encode(), path)

        # Remove deleted items from storage
        for key in self._storage.list(""):
            normalized_key = os.path.normpath(key)
            parts = normalized_key.split(os.sep)
            if len(parts) >= 2:
                collection, id = parts[-2], parts[-1]
                if not self._data.get(collection, {}).get(id):
                    self._storage.delete(key)

    def _load(self: "Database") -> None:
        """
        Load the current state of the database from storage.

        This method iterates over all the paths in
        storage, extracting the
        collection and id from the path and loading
        the associated data from
        storage. It then populates the in-memory
        database with the loaded data.

        Returns:
            None
        """
        self._data = {}
        for key in self._storage.list(""):
            normalized_key = os.path.normpath(key)
            parts = normalized_key.split(os.sep)
            if len(parts) >= 2:
                collection, id = parts[-2], parts[-1]
                data = self._storage.load(key)
                if collection not in self._data:
                    self._data[collection] = {}
                self._data[collection][id] = json.loads(data.decode())
