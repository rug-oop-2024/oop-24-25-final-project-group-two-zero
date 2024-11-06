import streamlit as st
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
    SupportVectorRegression
)
from autoop.core.ml.model.classification import (
    KNearestNeighbors,
    StochasticGradient,
    TreeClassification,
    TextClassificationModel
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
        "SupportVectorRegression": SupportVectorRegression
    }

    CLASSIFICATION_MODELS = {
        "KNearestNeighbors": KNearestNeighbors,
        "StochasticGradient": StochasticGradient,
        "DecisionTreeClassification": TreeClassification,
        "TextClassificationModel": TextClassificationModel
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

    FEATURE_TYPE_TO_EXTRACTOR = {
        'numerical': StandardScaler,
        'categorical': OneHotEncoder,
        # Add more mappings if needed
    }

    def __init__(self) -> None:
        """
        Initialize the Modelling class.
        """
        self.automl = AutoMLSystem.get_instance()
        self.automl.registry.refresh()
        self.datasets = self.automl.registry.list(type="dataset")

    # Function to get a model instance by name
    def get_model(self, model_name: str, task_type: str) -> Model:
        """
        Get a model instance by name.
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
        """
        dataset_options = {
            f"{artifact.name} (ID: {artifact.id})": artifact
            for artifact in self.datasets
        }
        return dataset_options

    def select_dataset(self, dataset_options: Dict[str, 'Artifact']) -> 'Artifact':
        """
        Select a dataset from the list of available datasets.
        """
        selected_dataset_name = st.selectbox('Choose a dataset:', list(dataset_options.keys()))
        selected_dataset = dataset_options[selected_dataset_name]
        return selected_dataset

    def choosing_input_output_features(self, feature_names: List[str]) -> Tuple[List[str], str]:
        """
        Allow user to select input features and target feature.
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

    def model_selection(self, target_feature_selected: Feature, input_features_selected: List[Feature]) -> Tuple[str, str]:
        """
        Select a model based on the task type (classification or regression).
        """
        # Determine if the target feature is categorical or numerical
        target_feature_type = target_feature_selected.type

        if target_feature_type == "categorical":
            task_type = "classification"
            st.write("Target feature is categorical, so this is a classification task.")
            available_models = self.CLASSIFICATION_MODELS
        else:
            task_type = "regression"
            st.write("Target feature is numerical, so this is a regression task.")
            available_models = self.REGRESSION_MODELS

        selected_model_name = st.selectbox(f"Select a {task_type} model:", list(available_models.keys()))
        return task_type, selected_model_name

    def select_metrics(self, task_type: str) -> Tuple[List['Metric'], List[str]]:
        """
        Allow user to select metrics based on task type.
        """
        available_metrics = self.get_metrics(task_type)
        selected_metric_names = st.multiselect(f"Select metrics for {task_type}", list(available_metrics.keys()))

        # Convert selected metric names to Metric instances
        selected_metrics = [available_metrics[name] for name in selected_metric_names]
        return selected_metrics, selected_metric_names

    def select_split_ratio(self) -> float:
        """
        Allow user to select the train-test split ratio.
        """
        split_ratio = st.slider('Select train-test split ratio', 0.1, 0.9, 0.8)
        return split_ratio

    def display_pipeline_summary(self, selected_model_name: str, split_ratio: float,
                                 selected_metric_names: List[str], input_features_selected_names: List[str],
                                 target_feature_selected_name: str) -> None:
        """
        Display a summary of the pipeline configuration.
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
        """
        st.write("## Results")
        for metric, value in results["metrics"]:
            st.write(f"{metric.name}: {value}")

    def display_model_explanations(self, results: dict) -> None:
        """
        Display the model explanations, such as confusion matrix and SHAP plots.
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
        Allow the user to specify one hyperparameter explicitly and specify acceptable ranges for others.

        Args:
            model_instance (Model): The model instance.

        Returns:
            dict: A dictionary containing selected hyperparameters and acceptable ranges.
        """
        st.write("### Model Hyperparameters")
        hyperparameters = model_instance.available_hyperparameters

        # List of hyperparameter names (excluding '_options' keys)
        hyperparameter_names = [param for param in hyperparameters.keys() if not param.endswith('_options')]

        # Let the user select one hyperparameter to specify explicitly
        selected_hyperparameter = st.selectbox("Select one hyperparameter to specify explicitly", hyperparameter_names)

        selected_hyperparameters = {}
        acceptable_ranges = {}

        for param in hyperparameter_names:
            default = hyperparameters[param]

            if param == selected_hyperparameter:
                st.write(f"**Specify a specific value for '{param}':**")
                # Prompt the user to enter a specific value
                if f"{param}_options" in hyperparameters:
                    options = hyperparameters[f"{param}_options"]
                    value = st.selectbox(f"{param}", options=options, index=options.index(default) if default in options else 0)
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
            else:
                st.write(f"**Specify acceptable range/options for '{param}':**")
                # Prompt the user to specify acceptable ranges or options
                if f"{param}_options" in hyperparameters:
                    options = hyperparameters[f"{param}_options"]
                    acceptable_values = st.multiselect(f"Acceptable options for {param}", options=options, default=default)
                    acceptable_ranges[param] = acceptable_values
                elif isinstance(default, int) or isinstance(default, float):
                    min_value = st.number_input(f"Minimum acceptable value for {param}", value=default)
                    max_value = st.number_input(f"Maximum acceptable value for {param}", value=default)
                    acceptable_ranges[param] = (min_value, max_value)
                elif isinstance(default, bool):
                    value = st.checkbox(f"Acceptable value for {param}", value=default)
                    acceptable_ranges[param] = value
                elif isinstance(default, str):
                    value = st.text_input(f"Acceptable value for {param}", value=default)
                    acceptable_ranges[param] = value
                else:
                    value = st.text_input(f"Acceptable value for {param}", value=str(default))
                    acceptable_ranges[param] = value

        # Combine selected hyperparameters and acceptable ranges
        hyperparameters_result = {
            'selected_hyperparameters': selected_hyperparameters,
            'acceptable_ranges': acceptable_ranges
        }
        return hyperparameters_result


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

        # Convert selected feature names to Feature instances
        input_features_selected = [feature_dict[name] for name in input_features_selected_names]
        target_feature_selected = feature_dict[target_feature_selected_name]
        target_feature_selected.is_target = True  # Mark the target feature

        # Mark input features as not target
        for feature in input_features_selected:
            feature.is_target = False

        # Step 3: Automatically select feature extractors for each input feature
        feature_extractors_selected = {}
        for feature in input_features_selected:
            feature_type = feature.type
            extractor_class = self.FEATURE_TYPE_TO_EXTRACTOR.get(feature_type)
            if extractor_class:
                extractor_instance = extractor_class()
                feature_extractors_selected[feature.name] = extractor_instance
                st.write(f"Automatically selected {extractor_class.__name__} for feature '{feature.name}' of type '{feature_type}'.")
            else:
                st.warning(f"No extractor found for feature type '{feature_type}' of feature '{feature.name}'")

        if input_features_selected and target_feature_selected:
            st.write(
                f"You selected input features: {input_features_selected_names}, "
                f"and target feature: {target_feature_selected_name}"
            )

            # Step 4: Select a model based on the task type
            task_type, selected_model_name = self.model_selection(target_feature_selected, input_features_selected)

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
            model_instance = self.get_model(selected_model_name, task_type)

            # Step 9: Select hyperparameters
            hyperparameters_result = self.select_model_hyperparameters(model_instance)
            selected_hyperparameters = hyperparameters_result['selected_hyperparameters']
            acceptable_ranges = hyperparameters_result['acceptable_ranges']

            # Re-initialize the model instance with selected hyperparameters
            model_class = type(model_instance)
            model_instance = model_class(**selected_hyperparameters)

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
        """
        self.run()


if __name__ == "__main__":
    app = Modelling()
    app.run()
