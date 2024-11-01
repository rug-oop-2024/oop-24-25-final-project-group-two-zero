from abc import ABC, abstractmethod
import os
from typing import List, Union
from glob import glob

class NotFoundError(Exception):
    def __init__(self, path):
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


class LocalStorage(Storage):

    def __init__(self, base_path: str="./assets"):
        self._base_path = base_path
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str):
        path = self._join_path(key)
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

    def delete(self, key: str="/"):
        self._assert_path_exists(self._join_path(key))
        path = self._join_path(key)
        os.remove(path)

    def list(self, prefix: str) -> List[str]:
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        # Use os.path.join for cross-platform compatibility
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        # Filter to keep only files
        files = [f for f in keys if os.path.isfile(f)]
        # Get paths relative to the base path
        relative_keys = [os.path.relpath(f, self._base_path) for f in files]
        return relative_keys


    def _assert_path_exists(self, path: str):
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        return os.path.join(self._base_path, path)
