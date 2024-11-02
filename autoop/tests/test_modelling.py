import unittest
from unittest.mock import patch, MagicMock
import streamlit as st
from autoop.core.ml.model.regression import LinearRegressionModel
from autoop.core.ml.model.classification import KNearestNeighbors
from app.pages.two_Modelling import Modelling

class TestModelling(unittest.TestCase):

    def setUp(self):
        self.modelling = Modelling()

    @patch('streamlit.set_page_config')
    @patch('streamlit.write')
    @patch('streamlit.stop')
    def test_run_no_datasets(self, mock_stop, mock_write, mock_set_page_config):
        self.modelling.datasets = []
        self.modelling.run()
        mock_write.assert_called_with("No datasets available, please upload a dataset first on the datasets page.")
        mock_stop.assert_called_once()

    @patch('streamlit.set_page_config')
    @patch('streamlit.write')
    @patch('streamlit.selectbox')
    @patch('streamlit.multiselect')
    @patch('streamlit.slider')
    @patch('streamlit.button')
    @patch('streamlit.text_input')
    @patch('app.pages.two_Modelling.Dataset.from_artifact')
    @patch('app.pages.two_Modelling.detect_feature_types')
    def test_run_with_datasets(self, mock_detect_feature_types, mock_from_artifact, mock_text_input, mock_button, mock_slider, mock_multiselect, mock_selectbox, mock_write, mock_set_page_config):
        mock_selectbox.side_effect = ['Test Dataset (ID: 1)', 'Feature1', 'LinearRegression']
        mock_multiselect.side_effect = [['Feature1', 'Feature2'], ['Mean Squared Error']]
        mock_slider.return_value = 0.8
        mock_button.side_effect = [True, True]
        mock_text_input.side_effect = ['Pipeline Name', '1.0.0']

        artifact_mock = MagicMock()
        artifact_mock.name = 'Test Dataset'
        artifact_mock.id = 1
        self.modelling.datasets = [artifact_mock]

        dataset_mock = MagicMock()
        mock_from_artifact.return_value = dataset_mock

        feature_mock = MagicMock()
        feature_mock.name = 'Feature1'
        feature_mock.type = 'numerical'
        mock_detect_feature_types.return_value = [feature_mock]

        self.modelling.run()

        mock_write.assert_any_call("Training the model...")
        mock_write.assert_any_call("## Results")
        mock_write.assert_any_call("## 💾 Save the Pipeline")
        mock_write.assert_any_call("Pipeline 'Pipeline Name' version '1.0.0' has been saved.")

    def test_get_model_regression(self):
        model = self.modelling.get_model("LinearRegression", "regression")
        self.assertIsInstance(model, LinearRegressionModel)

    def test_get_model_classification(self):
        model = self.modelling.get_model("KNearestNeighbors", "classification")
        self.assertIsInstance(model, KNearestNeighbors)

    def test_get_model_invalid(self):
        with self.assertRaises(ValueError):
            self.modelling.get_model("InvalidModel", "invalid")

    def test_get_metrics_regression(self):
        metrics = self.modelling.get_metrics("regression")
        self.assertIn("Mean Squared Error", metrics)

    def test_get_metrics_classification(self):
        metrics = self.modelling.get_metrics("classification")
        self.assertIn("Accuracy", metrics)

    def test_get_metrics_invalid(self):
        with self.assertRaises(ValueError):
            self.modelling.get_metrics("invalid")

if __name__ == '__main__':
    unittest.main()