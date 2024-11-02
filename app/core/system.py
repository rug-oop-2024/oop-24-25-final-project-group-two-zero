import os
from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry():
    def __init__(self, 
                 database: Database,
                 storage: Storage) -> None:
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
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
        self._database.set(f"artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
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
                id=id
            )
            artifacts.append(artifact)
        return artifacts


    def get(self, artifact_id: str) -> Artifact:
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

    def delete(self, artifact_id: str):
        data = self._database.get("artifacts", artifact_id)
        asset_path = os.path.normpath(data["asset_path"])
        self._storage.delete(asset_path)
        self._database.delete("artifacts", artifact_id)
    
    def refresh(self) -> None:
        """Refresh the registry by reloading data from the database."""
        self._database.refresh()



class AutoMLSystem:
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    # Do something here I think
    @staticmethod
    def get_instance() -> "AutoMLSystem":
        if AutoMLSystem._instance is None:
            # Normalize base paths
            object_storage_path = os.path.normpath("./assets/objects")
            database_storage_path = os.path.normpath("./assets/dbo")

            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage(object_storage_path), 
                Database(
                    LocalStorage(database_storage_path)
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        return self._registry

