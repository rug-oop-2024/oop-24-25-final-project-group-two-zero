import streamlit as st
import numpy as np
import base64
from typing import List, Tuple, Dict
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
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
    """
    Modelling class for designing and training machine learning pipelines.
    """

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

    FEATURE_EXTRACTORS = {
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "OneHotEncoder": OneHotEncoder,
    }

    def __init__(self) -> None:
        """
        Initialize the Modelling class.

        This method initializes the AutoMLSystem instance, refreshes the artifact registry,
        and retrieves a list of all datasets in the system.

        The datasets are stored in the `datasets` attribute of the Modelling instance.
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
    def get_metrics(self, task_type: str) -> Dict[str, 'Metric']:
        """
        Get a dictionary of available metrics based on task type.

        Args:
            task_type (str): The task type of the model (either "regression" or "classification").

        Returns:
            dict: A dictionary of available metrics for the task type.

        Raises:
            ValueError: If the task type is unknown.
        """
        if task_type == "regression":
            return self.REGRESSION_METRICS
        elif task_type == "classification":
            return self.CLASSIFICATION_METRICS
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _mapping_display_name(self) -> Dict[str, 'Artifact']:
        """
        Map display names to Artifact instances.

        Returns:
            dict: A dictionary mapping display names to Artifact instances.
        """
        dataset_options = {
            f"{artifact.name} (ID: {artifact.id})": artifact
            for artifact in self.datasets
        }
        return dataset_options

    def select_dataset(self, dataset_options: Dict[str, 'Artifact']) -> 'Artifact':
        """
        Select a dataset from the list of available datasets.

        Args:
            dataset_options (dict): A dictionary of dataset display names and Artifact instances.

        Returns:
            Artifact: The selected dataset artifact.
        """
        selected_dataset_name = st.selectbox('Choose a dataset:', list(dataset_options.keys()))
        selected_dataset = dataset_options[selected_dataset_name]
        return selected_dataset

    def choosing_input_output_features(self, feature_names: List[str]) -> Tuple[List[str], str]:
        """
        Allow user to select input features and target feature.

        Args:
            feature_names (List[str]): List of all feature names.

        Returns:
            Tuple[List[str], str]: A tuple containing the list of input feature names and the target feature name.
        """
        # Step 1: Select input features
        input_features_selected_names = st.multiselect("Select input features", feature_names)
        # Ensure that at least one input feature is selected
        if not input_features_selected_names:
            st.warning("Please select at least one input feature.")
            st.stop()
        # Step 2: Select target feature, excluding input features
        remaining_feature_names = [name for name in feature_names if name not in input_features_selected_names]
        if not remaining_feature_names:
            st.warning("No features left to select as target. Please select fewer input features.")
            st.stop()
        target_feature_selected_name = st.selectbox("Select target feature", remaining_feature_names)
        return input_features_selected_names, target_feature_selected_name

    def model_selection(self, target_feature_selected: Feature) -> Tuple[str, str]:
        """
        Select a model based on the task type (classification or regression).

        Args:
            target_feature_selected (Feature): The selected target feature.

        Returns:
            Tuple[str, str]: A tuple containing the task type and the selected model name.
        """
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
        return task_type, selected_model_name

    def select_metrics(self, task_type: str) -> Tuple[List['Metric'], List[str]]:
        """
        Allow user to select metrics based on task type.

        Args:
            task_type (str): The task type of the model.

        Returns:
            Tuple[List[Metric], List[str]]: A tuple containing the list of selected Metric instances and their names.
        
        We can do this and some sort of pattern
        """
        available_metrics = self.get_metrics(task_type)
        selected_metric_names = st.multiselect(f"Select metrics for {task_type}", list(available_metrics.keys()))

        # Convert selected metric names to Metric instances
        selected_metrics = [available_metrics[name] for name in selected_metric_names]
        return selected_metrics, selected_metric_names

    def select_split_ratio(self) -> float:
        """
        Allow user to select the train-test split ratio.

        Returns:
            float: The selected split ratio.
        """
        split_ratio = st.slider('Select train-test split ratio', 0.1, 0.9, 0.8)
        return split_ratio

    def display_pipeline_summary(self, selected_model_name: str, split_ratio: float,
                                 selected_metric_names: List[str], input_features_selected_names: List[str],
                                 target_feature_selected_name: str) -> None:
        """
        Display a summary of the pipeline configuration.

        Args:
            selected_model_name (str): The name of the selected model.
            split_ratio (float): The train-test split ratio.
            selected_metric_names (List[str]): The names of the selected metrics.
            input_features_selected_names (List[str]): The names of the selected input features.
            target_feature_selected_name (str): The name of the selected target feature.
        """
        st.write("## 📋 Pipeline Summary")
        st.markdown(f"""
        - **Selected Model**: `{selected_model_name}`
        - **Train-Test Split Ratio**: `{split_ratio}`
        - **Selected Metrics**: `{', '.join(selected_metric_names)}`
        - **Input Features**: `{', '.join(input_features_selected_names)}`
        - **Target Feature**: `{target_feature_selected_name}`
        """)

    def train_model(self, model_instance: Model, dataset_chosen: Dataset,
                    input_features_selected: List[Feature], target_feature_selected: Feature,
                    split_ratio: float, selected_metrics: List['Metric'],
                    feature_extractors_selected: Dict[str, str]) -> Pipeline:
        """
        Train the model using the specified configuration.

        Args:
            model_instance (Model): The model instance to train.
            dataset_chosen (Dataset): The selected dataset.
            input_features_selected (List[Feature]): List of selected input features.
            target_feature_selected (Feature): The selected target feature.
            split_ratio (float): The train-test split ratio.
            selected_metrics (List[Metric]): List of selected metrics.
            feature_extractors_selected (Dict[str, str]): Selected feature extractors for each input feature.

        Returns:
            Pipeline: The trained pipeline.
        """
        if st.button("Train Model"):
            st.write("Training the model...")

            # Build the pipeline with the selected model
            pipeline = Pipeline(
                metrics=selected_metrics,
                dataset=dataset_chosen,
                model=model_instance,
                input_features=input_features_selected,
                target_feature=target_feature_selected,
                split=split_ratio,
                feature_extractors=feature_extractors_selected,
            )

            # Execute the pipeline (train and evaluate the model)
            results = pipeline.execute()

            # Display results and explanations
            self.display_results(results)
            self.display_model_explanations(results)

            # Store the trained pipeline in session state
            st.session_state['trained_pipeline'] = pipeline
            return pipeline
        else:
            st.warning("Please click 'Train Model' to proceed.")
            return None

    def display_results(self, results: dict) -> None:
        """
        Display the evaluation metrics.

        Args:
            results (dict): The results dictionary from pipeline execution.
        """
        st.write("## Results")
        for metric, value in results["metrics"]:
            st.write(f"{metric.name}: {value}")

    def display_model_explanations(self, results: dict) -> None:
        """
        Display the model explanations, such as confusion matrix and SHAP plots.

        Args:
            results (dict): The results dictionary from pipeline execution.
        """
        report = results.get("report", {})
        if report:
            st.write("## Model Explanation")
            # Display Confusion Matrix if available
            confusion_matrix_png = report.get('confusion_matrix')
            if confusion_matrix_png:
                st.write("### Confusion Matrix")
                st.image(base64.b64decode(confusion_matrix_png), format='PNG')

            # Display SHAP Summary Plot
            shap_summary_png = report.get('shap_summary')
            if shap_summary_png:
                st.write("### SHAP Summary Plot")
                st.image(base64.b64decode(shap_summary_png), format='PNG')

            # Display Sensitivity Analysis
            sensitivities = report.get('sensitivity_results')
            if sensitivities:
                st.write("### Sensitivity Analysis")
                sensitivity_df = pd.DataFrame(list(sensitivities.items()), columns=['Feature', 'Sensitivity'])
                st.bar_chart(sensitivity_df.set_index('Feature'))


    def save_pipeline(self, pipeline: Pipeline) -> None:
        """
        Save the trained pipeline as an artifact.

        Args:
            pipeline (Pipeline): The trained pipeline.
        """
        if pipeline:
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

    def select_model_hyperparameters(self, model_instance: Model) -> dict:
        """
        Allow user to select hyperparameters for the selected model.

        Args:
            model_instance (Model): The model instance.

        Returns:
            dict: The hyperparameters selected by the user.
        """
        st.write("### Model Hyperparameters")
        hyperparameters = model_instance.available_hyperparameters
        selected_hyperparameters = {}
        for param, default in hyperparameters.items():
            if isinstance(default, list):
                value = st.selectbox(f"{param}", options=default, index=default.index(default) if default in default else 0)
            elif isinstance(default, bool):
                value = st.checkbox(f"{param}", value=default)
            elif isinstance(default, int):
                value = st.number_input(f"{param}", value=default, step=1)
            elif isinstance(default, float):
                value = st.number_input(f"{param}", value=default)
            elif isinstance(default, str):
                value = st.text_input(f"{param}", value=default)
            else:
                value = st.text_input(f"{param}", value=str(default))
            selected_hyperparameters[param] = value
        return selected_hyperparameters

    def run(self) -> None:
        """
        Main method to run the Modelling page.
        """
        st.set_page_config(page_title="Modelling", page_icon="📈")
        st.write("# ⚙ Modelling")
        st.write("In this section, you can design a machine learning pipeline to train a model on a dataset.")

        if not self.datasets:
            st.write("No datasets available, please upload a dataset first on the datasets page.")
            st.stop()

        # Map display names to Artifact instances
        dataset_options = self._mapping_display_name()

        # Step 1: Select dataset
        selected_dataset = self.select_dataset(dataset_options)

        # Convert the Artifact to a Dataset
        dataset_chosen = Dataset.from_artifact(selected_dataset)

        # Detect features
        feature_list = detect_feature_types(dataset_chosen)  # Returns a list of Feature instances
        feature_names = [feature.name for feature in feature_list]

        # Map feature names to Feature instances for easy lookup
        feature_dict = {feature.name: feature for feature in feature_list}

        # Step 2: Select input features and target feature
        input_features_selected_names, target_feature_selected_name = self.choosing_input_output_features(feature_names)

        # Step 3: Select feature extractors for each input feature
        feature_extractors_selected = {}
        for feature_name in input_features_selected_names:
            extractor_name = st.selectbox(
                f"Select feature extractor for {feature_name}",
                list(self.FEATURE_EXTRACTORS.keys()),
                key=f"extractor_{feature_name}"
            )
            feature_extractors_selected[feature_name] = extractor_name

        # Convert selected feature names to Feature instances
        input_features_selected = [feature_dict[name] for name in input_features_selected_names]
        target_feature_selected = feature_dict[target_feature_selected_name]
        target_feature_selected.is_target = True  # Mark the target feature

        # Mark input features as not target
        for feature in input_features_selected:
            feature.is_target = False

        if input_features_selected and target_feature_selected:
            st.write(
                f"You selected input features: {input_features_selected_names}, "
                f"and target feature: {target_feature_selected_name}"
            )

            # Step 4: Select a model based on the task type
            task_type, selected_model_name = self.model_selection(target_feature_selected)

            # Step 5: Select dataset split ratio
            split_ratio = self.select_split_ratio()

            # Step 6: Select metrics
            selected_metrics, selected_metric_names = self.select_metrics(task_type)

            # Step 7: Display a pipeline summary
            self.display_pipeline_summary(
                selected_model_name, split_ratio, selected_metric_names,
                input_features_selected_names, target_feature_selected_name
            )

            # Step 8: Get the model instance
            model_class = self.get_model_class(selected_model_name, task_type)
            model_instance = model_class()

            # Step 9: Select hyperparameters
            hyperparameters = self.select_model_hyperparameters(model_instance)
            # Re-initialize the model instance with selected hyperparameters
            model_instance = model_class(**hyperparameters)

            # Step 10: Train the model
            pipeline = self.train_model(
                model_instance=model_instance,
                dataset_chosen=dataset_chosen,
                input_features_selected=input_features_selected,
                target_feature_selected=target_feature_selected,
                split_ratio=split_ratio,
                selected_metrics=selected_metrics,
                feature_extractors_selected=feature_extractors_selected
            )

            # Step 11: Save the pipeline as an artifact
            self.save_pipeline(pipeline)

    def starter_modelling_page(self) -> None:
        """
        This is a placeholder method to start the Modelling page with a fresh UI.

        It simply calls the run method to start the page.
        """
        self.run()


if __name__ == "__main__":
    app = Modelling()
    app.run()
