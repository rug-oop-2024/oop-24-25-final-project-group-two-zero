

# dataset.py
from autoop.core.ml.artifact import Artifact
from abc import ABC, abstractmethod
import pandas as pd
import io
import h5py
import json
import os

class Dataset(Artifact):
    """
    Create something that automatically fills in the different
    
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes a Dataset instance.

        This constructor is used to initialize a dataset artifact instance
        with the specified arguments and keyword arguments.
        This has the arguments:
            name,
            asset_path,
            data.to_csv(index=False).encode(),
            version,
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
        version: str = "1.0.0"
        ) -> "Dataset":
        '''
        Creates a dataset artifact from a pandas DataFrame
        This is done
        '''
        if not isinstance(data, pd.DataFrame):
            return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.encode(),
            version=version,
            )
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )
        
    def save(self, directory: str = 'assets/objects/datasets'):
        """
        I think I have to put this into the objects or I should put this in a database
        I'm still not sure



        Saves the Artifact instance to an HDF5 file of a dataset.

        Parameters:
            directory (str): The directory where the file will be saved.

        # I have to change the directory of this to enter assets,
        instead of create something completely new
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f'{self.id}.h5')
        with h5py.File(file_path, 'w') as h5f:
            # Store binary data
            h5f.create_dataset('data', data=self.data)
            # Store attributes
            h5f.attrs['name'] = self.name
            h5f.attrs['asset_path'] = self.asset_path
            h5f.attrs['version'] = self.version
            h5f.attrs['type'] = self.type
            h5f.attrs['tags'] = json.dumps(self.tags)  # Convert list to JSON string
            h5f.attrs['id'] = self.id
            # Store metadata as JSON string
            h5f.attrs['metadata'] = json.dumps(self.metadata)

    @classmethod
    def read(cls, id: str, directory: str = 'dataset'):
        """
        Or maybe 

        Reads the Artifact from an HDF5 file and recreates the Artifact instance.


        Parameters:
            id (str): The unique identifier of the artifact.
            directory (str): The directory where the file is located.

        Returns:
            Artifact: An instance of the Artifact class.
        """
        file_path = os.path.join(directory, f'{id}.h5')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found for id {id}")
        with h5py.File(file_path, 'r') as h5f:
            # Read data
            data = h5f['data'][()]
            # Read attributes
            name = h5f.attrs['name']
            asset_path = h5f.attrs['asset_path']
            version = h5f.attrs['version']
            type_ = h5f.attrs['type']
            tags = json.loads(h5f.attrs['tags'])  # Convert JSON string back to list
            id_ = h5f.attrs['id']
            # Read metadata
            metadata = json.loads(h5f.attrs['metadata'])  # Convert JSON string back to dict
            # Create Dataset instance
            return cls(
                name=name,
                asset_path=asset_path,
                data=data,
                version=version,
                type=type_,
                tags=tags,
                metadata=metadata,
                id=id_
            )
    
    def __str__(self):
        return f"data: {self.data}"

