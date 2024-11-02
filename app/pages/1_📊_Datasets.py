# app/pages/datasets.py
import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
import os

class Starting:
    def __init__(self):
        self.automl = AutoMLSystem.get_instance()
        self.datasets_list = []
        self.name = None

    def name_dataset(self):
        name_dataset = st.text_input('Enter the dataset name', '')
        if not name_dataset:
            st.warning('Please enter a dataset name.')
            st.stop()
        self.name = name_dataset

    def upload_dataset(self):
        self.name_dataset()
        uploaded_file = st.file_uploader("Choose a dataset file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                st.write("Uploaded Dataset:")
                st.dataframe(df)

                # Use os.path.join and os.path.normpath
                asset_path = os.path.join("datasets", uploaded_file.name)
                asset_path = os.path.normpath(asset_path)

                dataset = Dataset.from_dataframe(
                    data=df,
                    name=self.name,
                    asset_path=asset_path,
                    version="1.0.0"
                )
                self.automl.registry.register(dataset)
                st.write("Dataset successfully uploaded and processed.")
            else:
                st.write("Please upload a CSV file.")
        else:
            st.write("No file uploaded.")


    def available_datasets(self):
        # Refresh the datasets list
        self.datasets_list = self.automl.registry.list(type="dataset")
        if self.datasets_list:
            st.write("Available Datasets:")
            # Create a dictionary mapping display names to artifact IDs
            dataset_options = {f"{artifact.name} (ID: {artifact.id})": artifact.id for artifact in self.datasets_list}
            selected_datasets = st.multiselect("Select datasets to remove:", list(dataset_options.keys()))
            if selected_datasets:
                if st.button("Remove Selected Datasets"):
                    for dataset_name in selected_datasets:
                        artifact_id = dataset_options[dataset_name]
                        self.automl.registry.delete(artifact_id)
                    st.success("Selected datasets have been removed.")
                    # Refresh the registry and datasets list after deletion
                    self.automl.registry.refresh()
                    self.datasets_list = self.automl.registry.list(type="dataset")
            else:
                st.write("Select one or more datasets to remove.")
        else:
            st.write("No datasets available.")


    def remove_dataset(self, dataset_name):
        # Refresh the datasets list
        self.datasets_list = self.automl.registry.list(type="dataset")
        if self.datasets_list:
            for artifact in self.datasets_list:
                if artifact.name == dataset_name:
                    self.automl.registry.delete(artifact.id)
                    st.write(f"Dataset '{dataset_name}' removed successfully.")
                    return
            st.write(f"Dataset '{dataset_name}' not found.")
        else:
            st.write("No datasets available.")



    def choose_to_upload(self):
        choice = st.radio("Choose an option:", ("Upload Dataset", "List/Remove Datasets"))
        if choice == "Upload Dataset":
            self.upload_dataset()
        elif choice == "List/Remove Datasets":
            self.available_datasets()


if __name__ == "__main__":
    start = Starting()
    start.choose_to_upload()
