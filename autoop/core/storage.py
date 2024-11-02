from abc import ABC, abstractmethod
import os
from typing import List, Union
from glob import glob

class NotFoundError(Exception):
    def __init__(self, path: str) -> None:
        super().__init__(f"Path not found: {path}")

class Storage(ABC):

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


# storage.py
import os

# storage.py
import os

class LocalStorage(Storage):
    def __init__(self, base_path: str = "./assets/objects"):
        self._base_path = os.path.abspath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def _join_path(self, key: str) -> str:
        # Normalize the key first
        key = os.path.normpath(key)
        # Join the base path and the key
        path = os.path.join(self._base_path, key)
        # Normalize the full path
        return os.path.normpath(path)

    def save(self, data: bytes, key: str):
        path = self._join_path(key)
        print(f"[LocalStorage] Saving data to: {path}")
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str):
        path = self._join_path(key)
        if os.path.exists(path):
            os.remove(path)
        else:
            print(f"Warning: The path '{path}' was not found. Skipping deletion.")

    def list(self, path: str) -> list:
        full_path = self._join_path(path)
        if os.path.exists(full_path):
            # Use glob to list files
            return [os.path.relpath(os.path.join(dp, f), self._base_path)
                    for dp, dn, filenames in os.walk(full_path)
                    for f in filenames]
        else:
            return []

    def _assert_path_exists(self, path: str):
        if not os.path.exists(path):
            raise NotFoundError(path)

