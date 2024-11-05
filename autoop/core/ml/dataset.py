from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize a Dataset instance.

        This method initializes a Dataset instance by calling the Artifact's
        constructor with the type set to "dataset".

        Args:
            *args: The positional arguments for the Artifact constructor.
            **kwargs: The keyword arguments for the Artifact constructor.
        """
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

        Args:
            data (pd.DataFrame): DataFrame containing the dataset.
            name (str): Name of the dataset.
            asset_path (str): Path where dataset is stored.
            version (str): Version of the dataset.

        Returns:
            Dataset: An instance of the Dataset class.
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

        Args:
            artifact (Artifact): Artifact instance to be converted.

        Returns:
            Dataset: An instance of the Dataset class.
        """
        if artifact.type != "dataset":
            raise ValueError("Artifact is not of type 'dataset'")
        return cls(
            name=artifact.name,
            asset_path=artifact.asset_path,
            data=artifact.data,
            version=artifact.version,
            tags=artifact.tags,
            metadata=artifact.metadata,
            id=artifact.id,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset data to a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame representation of the dataset.
        """
        if not self.data:
            raise ValueError("No data in dataset.")
        data_io = io.BytesIO(self.data)
        df = pd.read_csv(data_io)
        return df

    def __str__(self) -> str:
        """
        Returns a string representation of the Dataset instance.

        The representation includes the dataset's data in a byte format.

        Returns:
            str: A string that contains the data attribute of the Dataset.
        """
        return f"data: {self.data}"
