import unittest
from sklearn.datasets import load_iris, fetch_openml
import pandas as pd

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types

class TestFeatures(unittest.TestCase):

    def setUp(self) -> None:
        """
        Setup method for the test class.

        This method is called before each test, and is used to set up any state that
        is required for the tests to run. In this case, it does nothing, but it is
        included for clarity and consistency with other test classes.

        :return: None
        """
        pass

    def test_detect_features_continuous(self):
        """
        Test that detect_feature_types correctly identifies all features as numerical in a dataset
        containing only continuous data.

        :return: None
        """
        iris = load_iris()
        df = pd.DataFrame(
            iris.data,
            columns=iris.feature_names
        )
        dataset = Dataset.from_dataframe(
            name="iris",
            asset_path="iris.csv",
            data=df,
        )
        self.X = iris.data
        self.y = iris.target
        features = detect_feature_types(dataset)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 4)
        for feature in features:
            self.assertIsInstance(feature, Feature)
            self.assertEqual(feature.name in iris.feature_names, True)
            self.assertEqual(feature.type, "numerical")
        
    def test_detect_features_with_categories(self):
        """
        Test that detect_feature_types correctly identifies both numerical and categorical features
        in a dataset containing a mix of continuous and categorical data.

        :return: None
        """
        data = fetch_openml(name="adult", version=1, parser="auto")
        df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=df,
        )
        features = detect_feature_types(dataset)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 14)
        numerical_columns = [
            "age",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]
        categorical_columns = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        for feature in features:
            self.assertIsInstance(feature, Feature)
            self.assertEqual(feature.name in data.feature_names, True)
        for detected_feature in filter(lambda x: x.name in numerical_columns, features):
            self.assertEqual(detected_feature.type, "numerical")
        for detected_feature in filter(lambda x: x.name in categorical_columns, features):
            self.assertEqual(detected_feature.type, "categorical")
