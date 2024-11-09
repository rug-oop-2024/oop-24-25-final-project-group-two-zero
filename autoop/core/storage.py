import os
from abc import ABC, abstractmethod
from typing import List
from glob import glob


class NotFoundError(Exception):
    """Exception raised when a path is not found."""
    def __init__(self, path: str) -> None:
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """Abstract base class for storage."""
    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path.

        Args:
            data (bytes): Data to save.
            path (str): Path to save data.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path.

        Args:
            path (str): Path to load data from.

        Returns:
            bytes: Loaded data.
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path.

        Args:
            path (str): Path to delete data from.
        """
        pass

    @abstractmethod
    def list(self, path: str) -> List[str]:
        """
        List all paths under a given path.

        Args:
            path (str): Path to list.

        Returns:
            list: List of paths.
        """
        pass


class LocalStorage(Storage):
    def __init__(
            self,
            base_path: str = "./assets/objects"
        ):
        """
        Initialize the LocalStorage instance.

        Args:
            base_path (str): The base directory
                path for storage. Defaults to "./assets/objects".

        This constructor sets the base path
        for local storage, creating the directory if it does not exist.
        """
        self._base_path = os.path.abspath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def _join_path(self, key: str) -> str:
        """Joins the base path with a given key and normalizes the path."""
        key = os.path.normpath(key)
        path = os.path.join(self._base_path, key)
        return os.path.normpath(path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save data to a given path.

        Args:
            data (bytes): Data to save.
            path (str): Path to save data.
        returns:
            None
        """
        path = self._join_path(key)
        print(f"[LocalStorage] Saving data to: {path}")
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load data from a given path.

        Args:
            key (str): The key of the data to load.

        Returns:
            bytes: The loaded data.

        Raises:
            NotFoundError: If the specified
                path does not exist.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, "rb") as f:
            return f.read()

    def delete(self, key: str) -> None:
        """
        Delete data at a given path.

        Args:
            key (str): The key of the data to delete.

        Raises:
            NotFoundError: If the specified path does not exist.
        """
        path = self._join_path(key)
        if os.path.exists(path):
            os.remove(path)
        else:
            print(f"Warning: The path '{path}\
                   was not found. Skipping deletion.")

    def list(self, path: str) -> List[str]:
        """
        List all paths under a given path.

        Args:
            path (str): Path to list.

        Returns:
            list: List of paths.
        """
        full_path = self._join_path(path)
        if os.path.exists(full_path):
            return [
                os.path.relpath(os.path.join(dp, f), self._base_path)
                for dp, _, filenames in os.walk(full_path)
                for f in filenames
            ]
        return []

    def _assert_path_exists(self, path: str) -> None:
        """
        Asserts that a given path exists.

        Args:
            path (str): The path to assert.

        Raises:
            NotFoundError: If the path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)
