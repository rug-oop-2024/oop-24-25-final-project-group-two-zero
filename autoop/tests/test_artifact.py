import unittest
import os
import shutil
from autoop.core.ml.artifact import Artifact


class TestArtifact(unittest.TestCase):

    def setUp(self):
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
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_artifact_initialization(self):
        self.assertEqual(self.artifact.name, self.artifact_data['name'])
        self.assertEqual(self.artifact.asset_path, self.artifact_data['asset_path'])
        self.assertEqual(self.artifact.data, self.artifact_data['data'])
        self.assertEqual(self.artifact.version, self.artifact_data['version'])
        self.assertEqual(self.artifact.type, self.artifact_data['type'])
        self.assertEqual(self.artifact.tags, self.artifact_data['tags'])
        self.assertEqual(self.artifact.metadata, self.artifact_data['metadata'])
        self.assertIsNotNone(self.artifact.id)

    def test_artifact_save(self):
        self.artifact.save(directory=self.test_dir)
        file_path = os.path.join(self.test_dir, f'{self.artifact.id}.h5')
        self.assertTrue(os.path.exists(file_path))

    def test_artifact_read(self):
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
        with self.assertRaises(FileNotFoundError):
            Artifact.read(id='nonexistent_id', directory=self.test_dir)


if __name__ == '__main__':
    unittest.main()
