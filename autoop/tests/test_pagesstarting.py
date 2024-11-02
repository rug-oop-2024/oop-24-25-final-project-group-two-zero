import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import streamlit as st
from app.pages.one_Datasets import Starting

class TestStarting(unittest.TestCase):

    def setUp(self):
        self.starting = Starting()

    @patch('streamlit.text_input')
    @patch('streamlit.warning')
    @patch('streamlit.stop')
    def test_name_dataset_empty(self, mock_stop, mock_warning, mock_text_input):
        mock_text_input.return_value = ''
        self.starting.name_dataset()
        mock_warning.assert_called_once_with('Please enter a dataset name.')
        mock_stop.assert_called_once()

    @patch('streamlit.text_input')
    def test_name_dataset_valid(self, mock_text_input):
        mock_text_input.return_value = 'Test Dataset'
        self.starting.name_dataset()
        self.assertEqual(self.starting.name, 'Test Dataset')

    @patch('streamlit.file_uploader')
    @patch('streamlit.text_input')
    @patch('streamlit.write')
    @patch('pandas.read_csv')
    @patch('os.path.join')
    @patch('os.path.normpath')
    def test_upload_dataset_csv(self, mock_normpath, mock_join, mock_read_csv, mock_write, mock_text_input, mock_file_uploader):
        mock_text_input.return_value = 'Test Dataset'
        mock_file_uploader.return_value = MagicMock(name='test.csv')
        mock_read_csv.return_value = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_join.return_value = 'datasets/test.csv'
        mock_normpath.return_value = 'datasets/test.csv'

        with patch.object(self.starting.automl.registry, 'register') as mock_register:
            self.starting.upload_dataset()
            mock_register.assert_called_once()
            mock_write.assert_any_call("Dataset successfully uploaded and processed.")

    @patch('streamlit.file_uploader')
    @patch('streamlit.text_input')
    @patch('streamlit.write')
    def test_upload_dataset_no_file(self, mock_write, mock_text_input, mock_file_uploader):
        mock_text_input.return_value = 'Test Dataset'
        mock_file_uploader.return_value = None
        self.starting.upload_dataset()
        mock_write.assert_called_once_with("No file uploaded.")

    @patch('streamlit.multiselect')
    @patch('streamlit.button')
    @patch('streamlit.write')
    def test_available_datasets(self, mock_write, mock_button, mock_multiselect):
        mock_multiselect.return_value = ['Test Dataset (ID: 1)']
        mock_button.return_value = True

        artifact_mock = MagicMock()
        artifact_mock.name = 'Test Dataset'
        artifact_mock.id = 1
        self.starting.automl.registry.list = MagicMock(return_value=[artifact_mock])

        with patch.object(self.starting.automl.registry, 'delete') as mock_delete:
            self.starting.available_datasets()
            mock_delete.assert_called_once_with(1)
            mock_write.assert_any_call("Selected datasets have been removed.")

if __name__ == '__main__':
    unittest.main()