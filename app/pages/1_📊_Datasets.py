# this is for the dataset.py
import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

# Initialize AutoML system

class starting:

    def __init__(self):
        self.automl = AutoMLSystem.get_instance()
        self.datasets_list = self.automl.registry.list(type="dataset")
        self.name = None



    def available_datasets(self):
        # Display the available datasets in the Streamlit app
        if self.available_datasets:
            st.write("Available Datasets:")
            for dataset in self.datasets_list:
                st.write(dataset)
        else:
            st.write("No datasets available.")


    def name_dataset(self):
        name_dataset = st.text_area('Name of file', 'name of file')

        if not name_dataset or name_dataset == 'name of file':
            st.warning('Please type something')
            st.stop()  # This will stop the execution of the script
        self.name = name_dataset

        st.write('Your description is:', name_dataset)
        st.write('Hello, *%s*' % name_dataset)
    def upload_dataset(self):
        uploaded_file = st.file_uploader("Choose a dataset file", type=["csv", "xlsx"])
        if uploaded_file is not None:
        # Check the file extension and read it into a pandas DataFrame
            """
            @staticmethod
            def from_dataframe(
            data: pd.DataFrame,
            name: str,
            asset_path: str,
            version: str = "1.0.0"
            ) -> "Dataset":
            """
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)

                # Display the DataFrame in the Streamlit app
                st.write("Uploaded Dataset:")
                st.dataframe(df)

                # Convert the DataFrame to your Dataset format
                dataset = Dataset.from_dataframe(
                name= self.name,           # File name from user input
                # write down the asset path for me please codium
                asset_path= f"./assets/objects/datasets/{uploaded_file.name}",
                data=df,                     # The DataFrame
                version="1.0.0"              # Default version
                )
                dataset.save() # Save the dataset anyways as an object in the assets folder
                self.automl.registry.register(dataset)
                # Provide feedback in the Streamlit app
                st.write("Dataset successfully uploaded and processed.")
            else:
                st.write("Please upload a dataset.")

    def choose_to_upload(self):
        st.write("""
            What would you like to do,
            upload a dataset or list all available datasets?
        """)

        if st.button("Upload Dataset"):
            self.upload_dataset()
        elif st.button("List Available Datasets"):
            self.available_datasets()

starting().choose_to_upload()
