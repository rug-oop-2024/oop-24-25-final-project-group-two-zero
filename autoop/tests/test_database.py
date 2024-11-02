import unittest

from autoop.core.database import Database
from autoop.core.storage import LocalStorage
import random
import tempfile

class TestDatabase(unittest.TestCase):

    def setUp(self):
        """
        Set up a test database instance.

        Sets up a test database instance using a temporary directory for
        storage.

        """
        self.storage = LocalStorage(tempfile.mkdtemp())
        self.db = Database(self.storage)

    def test_init(self):
        """
        Test that the database instance is initialized correctly.

        Asserts that the database instance is of type Database.
        """
        self.assertIsInstance(self.db, Database)

    def test_set(self):
        """
        Test that data can be set in the database.

        Sets a key in the database with a random id and value, and then
        asserts that the value can be retrieved from the database.
        """
        id = str(random.randint(0, 100))
        entry = {"key": random.randint(0, 100)}
        self.db.set("collection", id, entry)
        self.assertEqual(self.db.get("collection", id)["key"], entry["key"])

    def test_delete(self):
        """
        Test that data can be deleted from the database.

        Sets a key in the database with a random id and value, deletes the
        key, and then asserts that the value can no longer be retrieved from
        the database. Also tests that the data is not retrieved even after
        calling refresh().
        """
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", id, value)
        self.db.delete("collection", id)
        self.assertIsNone(self.db.get("collection", id))
        self.db.refresh()
        self.assertIsNone(self.db.get("collection", id))

    def test_persistance(self):
        """
        Test that data is persisted between instances of the database.

        Sets a key in one instance of the database, creates another instance
        of the database with the same storage, and then asserts that the
        value can be retrieved from the other database instance.

        """
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", id, value)
        other_db = Database(self.storage)
        self.assertEqual(other_db.get("collection", id)["key"], value["key"])

    def test_refresh(self):
        """
        Test that calling refresh() updates the database instance with the latest data from storage.

        Sets a key in one instance of the database, creates another instance of the database with the same storage,
        calls refresh() on the other database instance, and then asserts that the value can be retrieved from the
        other database instance.

        """
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        other_db = Database(self.storage)
        self.db.set("collection", key, value)
        other_db.refresh()
        self.assertEqual(other_db.get("collection", key)["key"], value["key"])

    def test_list(self):
        """
        Test that the list() method returns all keys in the collection.

        Sets a key in the database, and then asserts that the key is in the
        list of keys returned by the list() method.
        """
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", key, value)
        # collection should now contain the key
        self.assertIn((key, value), self.db.list("collection"))