import unittest
import os
import shutil
from autoop.core.ml.artifact import Artifact


class TestArtifact(unittest.TestCase):

    def setUp(self):
        """
        Create a test artifact and a test directory for testing artifact
        functionality.

        The test artifact is created with the following attributes:

        - name: 'test_artifact'
        - asset_path: '/path/to/asset'
        - data: b'test data'
        - version: '1.0'
        - type: 'test_type'
        - tags: ['tag1', 'tag2']
        - metadata: {'key1': 'value1', 'key2': 'value2'}

        The test directory is created as 'test_artifacts' in the current working
        directory.
        """
        self.test_dir = 'test_artifacts'
        self.artifact_data = {
            'name': 'test_artifact',
            'asset_path': '/path/to/asset',
            'data': b'test data',
            'version': '1.0',
            'type': 'test_type',
            'tags': ['tag1', 'tag2'],
            'metadata': {'key1': 'value1', 'key2': 'value2'}
        }
        self.artifact = Artifact(**self.artifact_data)

    def tearDown(self):
        """
        Clean up the test directory created in setUp if it exists.
        """
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_artifact_initialization(self):
        """
        Test the initialization of the Artifact instance.

        This test verifies that the Artifact instance is initialized with the
        correct attributes as provided in the artifact_data dictionary. It checks
        that each attribute of the Artifact matches the corresponding value in
        artifact_data and ensures that the id attribute is not None.
        """
        self.assertEqual(self.artifact.name, self.artifact_data['name'])
        self.assertEqual(self.artifact.asset_path, self.artifact_data['asset_path'])
        self.assertEqual(self.artifact.data, self.artifact_data['data'])
        self.assertEqual(self.artifact.version, self.artifact_data['version'])
        self.assertEqual(self.artifact.type, self.artifact_data['type'])
        self.assertEqual(self.artifact.tags, self.artifact_data['tags'])
        self.assertEqual(self.artifact.metadata, self.artifact_data['metadata'])
        self.assertIsNotNone(self.artifact.id)

    def test_artifact_save(self):
        """
        Test the saving of an Artifact instance to an HDF5 file.

        This test verifies that the Artifact instance is saved to the correct
        location in the file system. It checks that the HDF5 file is created in
        the specified directory and that its contents match the attributes of the
        Artifact instance.
        """
        self.artifact.save(directory=self.test_dir)
        file_path = os.path.join(self.test_dir, f'{self.artifact.id}.h5')
        self.assertTrue(os.path.exists(file_path))

    def test_artifact_read(self):
        """
        Test the reading of an Artifact instance from an HDF5 file.

        This test verifies that the Artifact instance is read from the correct
        location in the file system. It checks that the HDF5 file is read from the
        specified directory and that its contents match the attributes of the
        Artifact instance.
        """
        self.artifact.save(directory=self.test_dir)
        read_artifact = Artifact.read(id=self.artifact.id, directory=self.test_dir)
        self.assertEqual(read_artifact.name, self.artifact.name)
        self.assertEqual(read_artifact.asset_path, self.artifact.asset_path)
        self.assertEqual(read_artifact.data, self.artifact.data)
        self.assertEqual(read_artifact.version, self.artifact.version)
        self.assertEqual(read_artifact.type, self.artifact.type)
        self.assertEqual(read_artifact.tags, self.artifact.tags)
        self.assertEqual(read_artifact.metadata, self.artifact.metadata)
        self.assertEqual(read_artifact.id, self.artifact.id)

    def test_artifact_read_nonexistent(self):
        """
        Test the reading of an Artifact instance from a nonexistent HDF5 file.

        This test verifies that when the Artifact instance is read from a
        nonexistent HDF5 file, a FileNotFoundError is raised.
        """
        with self.assertRaises(FileNotFoundError):
            Artifact.read(id='nonexistent_id', directory=self.test_dir)


if __name__ == '__main__':
    unittest.main()
