from __future__ import annotations
import os
from typing import List

from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import LocalStorage, Storage


class ArtifactRegistry:
    """Responsible for registering and listing artifacts in the registry."""

    def __init__(
        self: "ArtifactRegistry", database: Database, storage: Storage
    ) -> None:
        """
        Initialize an ArtifactRegistry instance.

        Args:
            database (Database): The database instance
                for storing artifact metadata.
            storage (Storage): The storage instance for saving artifact data.
        """
        self._database = database
        self._storage = storage

    def register(self: "ArtifactRegistry", artifact: Artifact) -> None:
        """
        Register an artifact in the registry.

        Args:
            artifact (Artifact): Artifact instance to be registered.
        """
        # Normalize the asset_path
        artifact.asset_path = os.path.normpath(artifact.asset_path)

        # Save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)

        # Save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self: "ArtifactRegistry", type: str = None) -> List[Artifact]:
        """
        List all artifacts in the registry.

        Args:
            type (str, optional): Filter artifacts by type.

        Returns:
            List[Artifact]: List of artifacts in the registry.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for artifact_id, data in entries:
            if type is not None and data["type"] != type:
                continue
            asset_path = os.path.normpath(data["asset_path"])
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=asset_path,
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(asset_path),
                type=data["type"],
                id=artifact_id,
            )
            artifacts.append(artifact)
        return artifacts

    def get(self: "ArtifactRegistry", artifact_id: str) -> Artifact:
        """
        Retrieve an artifact from the registry by its unique identifier.

        Args:
            artifact_id (str): Unique identifier of the artifact.

        Returns:
            Artifact: The retrieved artifact.

        Raises:
            KeyError: If the artifact does not exist in the registry.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self: "ArtifactRegistry", artifact_id: str) -> None:
        """
        Delete an artifact from the registry by its unique identifier.

        Args:
            artifact_id (str): Unique identifier of the artifact.

        Raises:
            KeyError: If the artifact does not exist in the registry.
        """
        data = self._database.get("artifacts", artifact_id)
        asset_path = os.path.normpath(data["asset_path"])
        self._storage.delete(asset_path)
        self._database.delete("artifacts", artifact_id)

    def refresh(self: "ArtifactRegistry") -> None:
        """
        Refresh the artifact registry by reloading the database.

        This method is useful if another process has modified the storage and
        you want to ensure the artifact registry is up to date. It will discard
        any unsaved changes you may have made to the artifact registry.
        """
        self._database.refresh()


class AutoMLSystem:
    """
    Manage the AutoML system.

    Implemented as a singleton to ensure only one instance exists.
    """

    _instance = None

    def __init__(
        self: "AutoMLSystem",
        storage: LocalStorage,
        database: Database,
    ) -> None:
        """
        Initialize an AutoMLSystem instance.

        Args:
            storage (LocalStorage): The storage instance
                for saving artifact data.
            database (Database): The database instance
                for storing artifact metadata.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> AutoMLSystem:
        """
        Return a singleton instance of the AutoMLSystem.

        This method ensures that only one instance of
        AutoMLSystem exists by using
        the singleton pattern. If an instance does not already exist,
        it initializes
        one with default storage and database paths,
        refreshes the database, and then
        returns the instance.

        Returns:
            AutoMLSystem: The singleton instance of the AutoMLSystem.
        """
        if AutoMLSystem._instance is None:
            # Normalize base paths
            object_storage_path = os.path.normpath("./assets/objects")
            database_storage_path = os.path.normpath("./assets/dbo")

            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage(object_storage_path),
                Database(LocalStorage(database_storage_path)),
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self: "AutoMLSystem") -> ArtifactRegistry:
        """
        Provide access to the artifact registry.

        Returns:
            ArtifactRegistry: The artifact registry instance.
        """
        return self._registry
