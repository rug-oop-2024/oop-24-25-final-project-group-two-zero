from autoop.core.ml.artifact import Artifact
from abc import ABC, abstractmethod
import pandas as pd
import io
import h5py
import json
import os


class Dataset(Artifact):
    def __init__(self, *args, **kwargs):
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame,
        name: str,
        asset_path: str,
        version: str = "1.0.0"
    ) -> "Dataset":
        """
        Creates a Dataset artifact from a pandas DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        data_bytes = data.to_csv(index=False).encode('utf-8')
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data_bytes,
            version=version,
        )
    @classmethod
    def from_artifact(cls, artifact: Artifact) -> "Dataset":
        """
        Reconstructs a Dataset instance from an Artifact instance.
        """
        if artifact.type != "dataset":
            raise ValueError("Artifact is not of type 'dataset'")
        return cls(
            name=artifact.name,
            asset_path=artifact.asset_path,
            data=artifact.data, # This is in bytes rn
            version=artifact.version,
            tags=artifact.tags,
            metadata=artifact.metadata,
            id=artifact.id,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset data to a pandas DataFrame.
        """
        if not self.data:
            raise ValueError("No data in dataset.")
        data_io = io.BytesIO(self.data)
        df = pd.read_csv(data_io)
        return df

    def __str__(self):
        return f"data: {self.data}"

