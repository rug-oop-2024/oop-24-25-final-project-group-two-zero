import unittest
import pandas as pd
from autoop.core.ml.dataset import Dataset


class TestDataset(unittest.TestCase):

    def setUp(self) -> None:
        """
        Sets up a test environment for the Dataset class.

        This method initializes a sample dataset using a pandas DataFrame
        with columns 'A', 'B', and 'C', and assigns it to the `data` attribute.
        It also sets up the attributes `name`, `asset_path`, and `version`
        for the dataset. Finally, it creates a Dataset instance using the
        from_dataframe method and assigns it to the `dataset` attribute.
        """
        self.data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        self.name = "test_dataset"
        self.asset_path = "test_dataset.csv"
        self.version = "1.0.0"
        self.dataset = Dataset.from_dataframe(
            data=self.data,
            name=self.name,
            asset_path=self.asset_path,
            version=self.version,
        )

    def test_from_dataframe(self):
        self.assertIsInstance(self.dataset, Dataset)
        self.assertEqual(self.dataset.name, self.name)
        self.assertEqual(self.dataset.asset_path, self.asset_path)
        self.assertEqual(self.dataset.version, self.version)
        self.assertIsNotNone(self.dataset.data)

    def test_to_dataframe(self):
        df = self.dataset.to_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        pd.testing.assert_frame_equal(df, self.data)

    def test_from_artifact(self):
        artifact = Dataset.from_artifact(self.dataset)
        self.assertIsInstance(artifact, Dataset)
        self.assertEqual(artifact.name, self.dataset.name)
        self.assertEqual(artifact.asset_path, self.dataset.asset_path)
        self.assertEqual(artifact.version, self.dataset.version)
        self.assertEqual(artifact.data, self.dataset.data)

    def test_str(self):
        dataset_str = str(self.dataset)
        self.assertIn("data:", dataset_str)

    def test_invalid_data_type(self):
        with self.assertRaises(ValueError):
            Dataset.from_dataframe(
                data="invalid_data",
                name=self.name,
                asset_path=self.asset_path,
                version=self.version,
            )

    def test_no_data_in_dataset(self):
        empty_dataset = Dataset(
            name=self.name, asset_path=self.asset_path, data=None, version=self.version
        )
        with self.assertRaises(ValueError):
            empty_dataset.to_dataframe()


if __name__ == "__main__":
    unittest.main()
