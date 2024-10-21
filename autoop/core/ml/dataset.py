from autoop.core.ml.artifact import Artifact
from abc import ABC, abstractmethod
import pandas as pd
import io

class Dataset(Artifact):

    def __init__(self, *args, **kwargs):
        """
        Initializes a Dataset instance.

        This constructor is used to initialize a dataset artifact instance
        with the specified arguments and keyword arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame,
        name: str,
        asset_path: str,
        version: str="1.0.0"
        ) -> "Dataset":
        '''
        Creates a dataset artifact from a pandas DataFrame
        '''
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )
        
    def read(self) -> pd.DataFrame:
        '''
        Reads the dataset from the stored artifact and returns a pandas DataFrame
        representation of it.
        '''
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))
    
    def save(self, data: pd.DataFrame) -> bytes:
        '''
        Saves the dataset to the artifact
        '''
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
    