# app/pages/datasets.py
import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

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

                dataset = Dataset.from_dataframe(
                    data=df,
                    name=self.name,
                    asset_path=f"datasets/{uploaded_file.name}",
                    version="1.0.0"
                )
                # Register the dataset using the ArtifactRegistry
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
            for artifact in self.datasets_list:
                st.write(f"Name: {artifact.name}, ID: {artifact.id}")
                # Convert artifact to dataset
                try:
                    dataset = Dataset.from_artifact(artifact)
                    # Display the dataset content
                    df = dataset.to_dataframe()
                    st.dataframe(df)
                except Exception as e:
                    st.write(f"Error loading dataset {artifact.name}: {e}")
        else:
            st.write("No datasets available.")



    def choose_to_upload(self):
        choice = st.radio("Choose an option:", ("Upload Dataset", "List Available Datasets"))
        if choice == "Upload Dataset":
            self.upload_dataset()
        elif choice == "List Available Datasets":
            self.available_datasets()

Starting().choose_to_upload()
