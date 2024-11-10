from typing import Any, Dict, List, Optional
import pandas as pd
from .artifact import Artifact
import io


class Dataset(Artifact):
    def __init__(
        self,
        name: str,
        asset_path: str,
        data: pd.DataFrame,
        version: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> None:
        """
        Initialize a Dataset instance.

        Args:
            name (str): Name of the dataset.
            asset_path (str): Path where dataset is stored.
            data (pd.DataFrame): The dataset as a pandas DataFrame.
            version (str): Version of the dataset.
            tags (List[str], optional): Tags for categorizing the dataset.
            metadata (Dict[str, Any], optional): Additional metadata.
            id (str, optional): Unique identifier.
        """

        # Convert DataFrame to bytes for storage
        super().__init__(
            name=name,
            asset_path=asset_path,
            data=data,  # Pass bytes to Artifact
            version=version,
            type="dataset",
            tags=tags,
            metadata=metadata,
            id=id,
        )

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
        if isinstance(data, pd.DataFrame) is False:
            raise ValueError("Data must be a pandas DataFrame.")
        return Artifact(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode("utf-8"),  # Pass DataFrame
            version=version,
            type="dataset",
        )

    @classmethod
    def from_artifact(cls, artifact: Artifact) -> "Dataset":
        """
        Reconstructs a Dataset instance from an Artifact instance.

        Args:
            artifact (Artifact):
                Artifact instance to be converted.

        Returns:
            Dataset: An instance of the Dataset class.
        """
        if artifact.type != "dataset":
            raise ValueError(
                "Artifact is not of type 'dataset'"
            )

        # Convert data bytes to DataFrame
        data_io: io.BytesIO = io.BytesIO(artifact.data)
        data_df: pd.DataFrame = pd.read_csv(data_io)

        return cls(
            name=artifact.name,
            asset_path=artifact.asset_path,
            data=data_df,  # Pass DataFrame
            version=artifact.version,
            tags=artifact.tags,
            metadata=artifact.metadata,
            id=artifact.id,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset data to a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame
                representation of the dataset.
        """
        if self.data is None:
            raise ValueError("No data in dataset.")
        elif isinstance(self.data, pd.DataFrame):
            # If self.data is already a DataFrame, return it directly
            return self.data
        else:
            # If self.data is bytes, convert it to a DataFrame
            data_io: io.BytesIO = io.BytesIO(self.data)
            df: pd.DataFrame = pd.read_csv(data_io)
            return df

    def read(self) -> pd.DataFrame:
        """
        Reads the dataset from the artifact
        and returns it as a pandas DataFrame.
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Saves the given DataFrame as bytes using the parent class's save method.

        Args:
            data (pd.DataFrame): The pandas DataFrame to be saved.

        Returns:
            bytes: The bytes representation of the saved data.
        """
        bytes: pd.DataFrame =\
            data.to_csv(index=False).encode()
        return super().save(bytes)

    def __str__(self) -> str:
        """
        Returns a string representation
        of the Dataset instance.
        """
        return f"""Dataset(name={self.name},
        version={self.version})"""


