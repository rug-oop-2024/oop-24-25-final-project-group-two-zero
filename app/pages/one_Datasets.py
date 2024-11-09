import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
import os


# Remove all the none rows from a dataset if there is any
class Starting:
    def __init__(self) -> None:
        """
        Initialize the Starting class.

        This method initializes the AutoML system and an empty list to
        store the available datasets. Additionally, it sets the dataset name
        to None.

        :return: None
        """
        self.automl = AutoMLSystem.get_instance()
        self.datasets_list = []
        self.name = None

    def name_dataset(self) -> None:
        """
        Prompt the user to enter a dataset name through a Streamlit text input.

        This method displays a text input field for the user to provide a dataset name.
        If no name is entered, a warning is displayed and the execution is stopped.
        Once a valid name is entered, it is stored in the `name` attribute of the class.

        :return: None
        """
        name_dataset = st.text_input("Enter the dataset name", "")
        if not name_dataset:
            st.warning("Please enter a dataset name.")
            st.stop()
        self.name = name_dataset

    def upload_dataset(self) -> None:
        """
        Upload a dataset file to the AutoML system.

        This method first calls `name_dataset` to prompt the user to enter a dataset name.
        Then, it displays a file uploader to the user to upload a dataset file.
        If the uploaded file is not a CSV file, a warning is displayed and the execution is stopped.
        If a valid CSV file is uploaded, it is processed and registered in the AutoML system.
        Finally, a success message is displayed to the user.

        :return: None
        """
        self.name_dataset()
        uploaded_file = st.file_uploader("Choose a dataset file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                st.write("Uploaded Dataset:")
                st.dataframe(df)

                # Register the dataset
                dataset = Dataset.from_dataframe(
                    data=df,
                    name=self.name,
                    # This is the asset path normalized
                    asset_path=os.path.normpath(
                        os.path.join("datasets", uploaded_file.name)
                    ),
                    version="1.0.0",
                )
                self.automl.registry.register(dataset)
                st.write("Dataset successfully uploaded and processed.")
            else:
                st.write("Please upload a CSV file.")
        else:
            st.write("No file uploaded.")

    def available_datasets(self) -> None:
        # Refresh the datasets list
        """
        Display a list of available datasets and allow the user to select one or more datasets to remove.

        This method first refreshes the list of available datasets in the AutoML system.
        Then, it displays the list of available datasets and prompts the user to select one or more
        datasets to remove. If the user selects one or more datasets and clicks the "Remove Selected
        Datasets" button, the selected datasets are removed from the AutoML system and the list of
        available datasets is refreshed.

        :return: None
        """
        self.datasets_list = self.automl.registry.list(type="dataset")
        if self.datasets_list:
            st.write("Available Datasets:")
            # Create a dictionary mapping display names to artifact IDs
            dataset_options = {
                f"{artifact.name} (ID: {artifact.id})": artifact.id
                for artifact in self.datasets_list
            }
            selected_datasets = st.multiselect(
                "Select datasets to remove:", list(dataset_options.keys())
            )
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

    def remove_dataset(self, dataset_name: str) -> None:
        # Refresh the datasets list
        """
        Remove a dataset from the AutoML system.

        This method first refreshes the list of available datasets in the AutoML system.
        Then, it looks for the dataset with the given name in the list of available datasets.
        If the dataset is found, it is removed from the AutoML system and a success message is displayed.
        If the dataset is not found, a message indicating that the dataset was not found is displayed.

        :param dataset_name: The name of the dataset to remove.
        :return: None
        """
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

    def choose_to_upload(self) -> None:
        """
        Prompts the user to choose between uploading a dataset and listing or removing existing datasets.

        This method displays a radio button to the user with two options: "Upload Dataset" and "List/Remove Datasets".
        Depending on the user's choice, it either calls the upload_dataset method or the available_datasets method.
        :return: None
        """
        choice = st.radio(
            "Choose an option:", ("Upload Dataset", "List/Remove Datasets")
        )
        if choice == "Upload Dataset":
            self.upload_dataset()
        elif choice == "List/Remove Datasets":
            self.available_datasets()


if __name__ == "__main__":
    start = Starting()
    start.choose_to_upload()
