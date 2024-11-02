import streamlit as st
import numpy as np
from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model.model import Model
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model.regression import (
    MultipleLinearRegression,
    RidgeRegression,
    LinearRegressionModel
)
from autoop.core.ml.model.classification import (
    KNearestNeighbors,
    StochasticGradient,
    TreeClassification
)
from autoop.core.ml.metric import (
    Accuracy,
    MeanSquaredError,
    MeanAbsoluteError,
    R2Score,
    Specificity,
    F1Score
)
from autoop.functional.feature import Feature, detect_feature_types


class Modelling:
    # Define the model lists for regression and classification
    REGRESSION_MODELS = {
        "MultipleLinearRegression": MultipleLinearRegression,
        "RidgeRegression": RidgeRegression,
        "LinearRegression": LinearRegressionModel
    }

    CLASSIFICATION_MODELS = {
        "StochasticGradient": StochasticGradient,
        "KNearestNeighbors": KNearestNeighbors,
        "DecisionTreeClassification": TreeClassification
    }

    # Define the metrics for regression and classification
    REGRESSION_METRICS = {
        "Mean Squared Error": MeanSquaredError(),
        "Mean Absolute Error": MeanAbsoluteError(),
        "R-Squared": R2Score(),
    }

    CLASSIFICATION_METRICS = {
        "Accuracy": Accuracy(),
        "F1 Score": F1Score(),
        "Specificity": Specificity(),
    }

    def __init__(self) -> None:
        """
        Initialize the Modelling class.

        This method initializes the AutoMLSystem instance, refreshes the artifact registry, and retrieves a list of all datasets in the system.

        The datasets are stored in the `datasets` attribute of the Modelling instance.

        :return: None
        """
        self.automl = AutoMLSystem.get_instance()
        self.automl.registry.refresh()
        self.datasets = self.automl.registry.list(type="dataset")

    # Function to get a model instance by name
    def get_model(self, model_name: str, task_type: str) -> Model:
        """
        Get a model instance by name.

        Args:
            model_name (str): The name of the model to retrieve.
            task_type (str): The task type of the model (either "regression" or "classification").

        Returns:
            Model: The model instance.

        Raises:
            ValueError: If the task type is unknown.
        """
        if task_type == "regression":
            return self.REGRESSION_MODELS.get(model_name)()
        elif task_type == "classification":
            return self.CLASSIFICATION_MODELS.get(model_name)()
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    # Function to get metrics based on task type
    def get_metrics(self, task_type: str) -> dict | None:
        """
        Get a dictionary of available metrics based on task type.

        Args:
            task_type (str): The task type of the model (either "regression" or "classification").

        Returns:
            dict | None: A dictionary of available metrics for the task type, or None if the task type is unknown.

        Raises:
            ValueError: If the task type is unknown.
        """
        if task_type == "regression":
            return self.REGRESSION_METRICS
        elif task_type == "classification":
            return self.CLASSIFICATION_METRICS
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def run(self) -> None:
        # Initialize Streamlit page
        """
        Initialize Streamlit page for the Modelling page.

        This page will allow users to design a machine learning pipeline to train a model on a dataset.

        The steps are as follows:

        1. Select a dataset from the list of available datasets
        2. Select input features and target feature from the dataset
        3. Select a model based on the task type (classification or regression)
        4. Select dataset split ratio
        5. Select metrics based on the task type
        6. Display a pipeline summary
        7. Train the model
        8. Save the pipeline as an artifact
        """
        st.set_page_config(page_title="Modelling", page_icon="📈")
        st.write("# ⚙ Modelling")
        st.write("In this section, you can design a machine learning pipeline to train a model on a dataset.")

        if not self.datasets:
            st.write("No datasets available, please upload a dataset first on the datasets page.")
            st.stop()

        # Map display names to Artifact instances
        dataset_options = {
            f"{artifact.name} (ID: {artifact.id})": artifact
            for artifact in self.datasets
        }

        # Allow user to select a dataset
        selected_dataset_name = st.selectbox('Choose a dataset:', list(dataset_options.keys()))
        selected_dataset = dataset_options[selected_dataset_name]

        # Convert the Artifact to a Dataset
        dataset_chosen = Dataset.from_artifact(selected_dataset)

        # Get features from the dataset
        detector = detect_feature_types()
        feature_list = detector(dataset_chosen)  # Returns a list of Feature instances
        feature_names = [feature.name for feature in feature_list]

        # Map feature names to Feature instances for easy lookup
        feature_dict = {feature.name: feature for feature in feature_list}

        # Step 1: Select input features and target feature
        input_features_selected_names = st.multiselect("Select input features", feature_names)
        target_feature_selected_name = st.selectbox("Select target feature", feature_names)

        # Convert selected feature names to Feature instances
        input_features_selected = [feature_dict[name] for name in input_features_selected_names]
        target_feature_selected = feature_dict[target_feature_selected_name]
        target_feature_selected.is_target = True  # Mark the target feature

        # Step 2: Detect task type (classification or regression)
        for feature in input_features_selected:
            feature.is_target = False

        if input_features_selected and target_feature_selected:
            st.write(
                f"You selected input features: {input_features_selected_names}, "
                f"and target feature: {target_feature_selected_name}"
            )

            # Determine if the target feature is categorical or numerical
            target_feature_type = target_feature_selected.type

            if target_feature_type == "categorical":
                task_type = "classification"
                st.write("Target feature is categorical, so this is a classification task.")
                available_models = list(self.CLASSIFICATION_MODELS.keys())
            else:
                task_type = "regression"
                st.write("Target feature is numerical, so this is a regression task.")
                available_models = list(self.REGRESSION_MODELS.keys())

            # Step 3: Allow the user to select a model based on the task type
            selected_model_name = st.selectbox(f"Select a {task_type} model:", available_models)

            # Step 4: Allow user to select dataset split ratio
            split_ratio = st.slider('Select train-test split ratio', 0.1, 0.9, 0.8)

            # Step 5: Allow user to select metrics based on task type
            available_metrics = self.get_metrics(task_type)
            selected_metric_names = st.multiselect(f"Select metrics for {task_type}", available_metrics.keys())

            # Convert selected metric names to Metric instances
            selected_metrics = [available_metrics[name] for name in selected_metric_names]

            # Step 6: Display a pipeline summary
            if selected_model_name and selected_metrics:
                st.write("## 📋 Pipeline Summary")
                st.markdown(f"""
                - **Selected Model**: `{selected_model_name}`
                - **Train-Test Split Ratio**: `{split_ratio}`
                - **Selected Metrics**: `{', '.join(selected_metric_names)}`
                - **Input Features**: `{', '.join(input_features_selected_names)}`
                - **Target Feature**: `{target_feature_selected_name}`
                """)

                # Step 7: Train the model
                if st.button("Train Model"):
                    st.write("Training the model...")

                    # Get the selected model instance
                    model_instance = self.get_model(selected_model_name, task_type)

                    # Build the pipeline with the selected model
                    pipeline = Pipeline(
                        metrics=selected_metrics,
                        dataset=dataset_chosen,
                        model=model_instance,
                        input_features=input_features_selected,
                        target_feature=target_feature_selected,
                        split=split_ratio
                    )

                    # Execute the pipeline (train and evaluate the model)
                    results = pipeline.execute()

                    # Report the selected metrics
                    st.write("## Results")
                    for metric, value in results["metrics"]:
                        st.write(f"{metric.name}: {value}")

                    # Store the trained pipeline in session state
                    st.session_state['trained_pipeline'] = pipeline

                # Step 8: Save the pipeline as an artifact
                st.write("## 💾 Save the Pipeline")
                pipeline_name = st.text_input("Enter a name for the pipeline")
                pipeline_version = st.text_input("Enter a version for the pipeline", "1.0.0")

                if st.button("Save Pipeline"):
                    if 'trained_pipeline' in st.session_state:
                        pipeline = st.session_state['trained_pipeline']
                        if pipeline_name:
                            # Convert the pipeline into an artifact
                            pipeline_artifact = pipeline.to_artifact(name=pipeline_name, version=pipeline_version)
                            # Register the pipeline artifact
                            self.automl.registry.register(pipeline_artifact)
                            st.success(
                                f"Pipeline '{pipeline_name}' version '{pipeline_version}' has been saved."
                            )
                        else:
                            st.warning("Please enter a name for the pipeline.")
                    else:
                        st.warning("Please train a model before saving the pipeline.")
            else:
                st.write("Please select a model and metrics to proceed.")

    def starter_modelling_page(self) -> None:
        """
        This is a placeholder method to start the Modelling page with a fresh UI.

        It simply calls the run method to start the page.
        """
        self.run()


if __name__ == "__main__":
    app = Modelling()
    app.run()
