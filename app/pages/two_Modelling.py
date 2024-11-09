import streamlit as st
import base64
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model.model import Model
from autoop.core.ml.dataset import Dataset
from sklearn.model_selection import train_test_split
from autoop.core.ml.model.regression import (
    MultipleLinearRegression,
    RidgeRegression,
    SupportVectorRegression,
)
from autoop.core.ml.model.classification import (
    KNearestNeighbors,
    StochasticGradient,
    TreeClassification,
    TextClassificationModel,
)
from autoop.core.ml.metric import (
    Accuracy,
    MeanSquaredError,
    MeanAbsoluteError,
    R2Score,
    Specificity,
    F1Score,
)
from sklearn.model_selection import GridSearchCV
from autoop.functional.feature import Feature, detect_feature_types
from autoop.core.ml.artifact import Artifact
import streamlit as st
import pandas as pd
import io
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import Metric
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.artifact import Artifact

st.set_page_config(page_title="Modelling", page_icon="📈")

"""This module provides a Streamlit interface for training a pipeline."""


def write_helper_text(text: str) -> None:
    """
    Write helper text in a specific style.

    Args:
        text (str): The text to be displayed.
    """
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine learning pipeline"
    "to train a model on a dataset."
)


class Modelling:
    """
    Unlike the other classes, there is no need ot have
    """

    REGRESSION_MODELS = {
        "MultipleLinearRegression": MultipleLinearRegression,
        "RidgeRegression": RidgeRegression,
        "SupportVectorRegression": SupportVectorRegression,
    }

    CLASSIFICATION_MODELS = {
        "KNearestNeighbors": KNearestNeighbors,
        "StochasticGradient": StochasticGradient,
        "DecisionTreeClassification": TreeClassification,
        "TextClassificationModel": TextClassificationModel,
    }

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
    scoring_options = {
        "Accuracy": "accuracy",
        "F1 Score": "f1",
        "Specificity": "roc_auc",
        "Mean Squared Error": "neg_mean_squared_error",
        "Mean Absolute Error": "neg_mean_absolute_error",
        "R-Squared": "r2",
    }

    def __init__(self) -> None:
        self.automl = AutoMLSystem.get_instance()
        self.datasets = self.automl.registry.list(type="dataset")

    def train_pipeline(
        self,
        dataset: Dataset,
        model: Model,
        input_features: List[Feature],
        target_feature: Feature,
        split_ratio: float,
        metrics_use: List["Metric"],
    ):
        pipeline = Pipeline(
            metrics_use, dataset, model, input_features, target_feature, split_ratio
        )
        # """
        # return {
        #     "metrics": self._metrics_results,
        #     "predictions": self._predictions,
        #     "report": self._report,
        # }
        # """
        values = pipeline.execute()
        print(values["predictions"])
        return pipeline

    def select_model_hyperparameters(self, model_instance: Model) -> dict:
        """
        Allow the user to specify acceptable ranges or options for all hyperparameters.

        Args:
            model_instance (Model): The model instance.

        Returns:
            dict: A dictionary containing acceptable ranges for hyperparameters.
        """
        st.write("### Model Hyperparameters")
        hyperparameters = model_instance.available_hyperparameters

        acceptable_ranges = {}

        for param, default in hyperparameters.items():
            if param.endswith("_options"):
                continue

            st.write(f"**Specify acceptable value(s) for '{param}':**")
            options_key = f"{param}_options"
            options = hyperparameters.get(options_key, None)

            if options:
                acceptable_values = st.multiselect(
                    f"Acceptable options for {param}", options=options, default=default
                )
                acceptable_ranges[param] = (
                    acceptable_values if acceptable_values else [default]
                )
            elif isinstance(default, bool):
                value = st.checkbox(f"Acceptable value for {param}", value=default)
                acceptable_ranges[param] = [value]
            elif isinstance(default, int):
                min_value = st.number_input(
                    f"Minimum acceptable value for {param}", value=default
                )
                max_value = st.number_input(
                    f"Maximum acceptable value for {param}", value=default + 10
                )
                acceptable_ranges[param] = list(
                    range(int(min_value), int(max_value) + 1)
                )
            elif isinstance(default, float):
                min_value = st.number_input(
                    f"Minimum acceptable value for {param}", value=default
                )
                max_value = st.number_input(
                    f"Maximum acceptable value for {param}", value=default + 1.0
                )
                acceptable_ranges[param] = np.linspace(
                    min_value, max_value, num=5
                ).tolist()
            elif default is None:
                # Handle parameters with default None
                value = st.text_input(
                    f"Acceptable value for {param} (enter 'None' or an integer)",
                    value="None",
                )
                if value.strip().lower() == "none":
                    acceptable_ranges[param] = [None]
                else:
                    try:
                        acceptable_ranges[param] = [int(value)]
                    except ValueError:
                        st.warning(
                            f"Invalid value for '{param}'. Please enter 'None' or an integer."
                        )
                        acceptable_ranges[param] = [None]
            elif isinstance(default, str):
                value = st.text_input(f"Acceptable value for {param}", value=default)
                acceptable_ranges[param] = [value]
            else:
                value = st.text_input(
                    f"Acceptable value for {param}", value=str(default)
                )
                acceptable_ranges[param] = [value]

        return acceptable_ranges

    def run(self):
        dataset_options = {f"{dataset.name}": dataset.id for dataset in self.datasets}

        selected_dataset_name = st.selectbox(
            "Select a dataset", list(dataset_options.keys())
        )

        # Get the actual dataset ID from the mapping
        dataset_id = dataset_options[selected_dataset_name]

        # Get the dataset artifact
        dataset_artifact = self.automl.registry.get(dataset_id)

        # Convert the Artifact to a Dataset
        dataset = Dataset.from_artifact(dataset_artifact)

        st.write(f"Selected dataset: {dataset.name}")

        df = dataset.to_dataframe()

        features = detect_feature_types(dataset)
        for feature in features:
            st.write(f"- {feature.name}: {feature.type}")

        st.header("Select the split ratio")
        split_ratio = st.slider(
            "Select the split ratio", min_value=0.0, max_value=1.0, value=0.8
        )

        st.header("Select the target feature")
        target_feature = st.selectbox(
            "Select the target feature", [feature for feature in features]
        )
        input_features = st.multiselect(
            "Select the input features",
            [feature for feature in features if feature != target_feature],
        )
        if not input_features:
            st.warning("Please select at least one input feature.")
            st.stop()
        input_feature_names = [feature.name for feature in input_features]
        target_feature_name = target_feature.name

        st.header("Select the model")
        if target_feature.type == "categorical":
            model_options = self.CLASSIFICATION_MODELS
        else:
            model_options = self.REGRESSION_MODELS
        st.header(f"these are the models {model_options.items()}")
        model_name_to_class = {name: cls for name, cls in model_options.items()}

        selected_model_name = st.selectbox(
            "Select the model", list(model_name_to_class.keys())
        )

        # Get the model class based on the selected name
        model_class = model_name_to_class[selected_model_name]

        # Create an instance of the model class
        model = model_class()

        st.header("Select the metrics")
        if target_feature.type == "categorical":
            metric_options = self.CLASSIFICATION_METRICS
        else:
            metric_options = self.REGRESSION_METRICS

        # Create a mapping from metric names to metric instances
        metric_name_to_instance = {
            name: instance for name, instance in metric_options.items()
        }
        # Use this for the hyperparameter selection
        # # Use the metric names in the multiselect
        selected_metric_names = st.multiselect(
            "Select the metrics", list(metric_name_to_instance.keys())
        )

        # # Check if there is a metric is selected
        if not selected_metric_names:
            st.warning("Please select at least one metric.")
            st.stop()
        # Create the scoring parameter for GridSearchCV
        if len(selected_metric_names) == 1:
            scoring = self.scoring_options.get(selected_metric_names[0], None)
            if scoring is None:
                st.warning(
                    f"Scoring for metric '{selected_metric_names[0]}' is not defined."
                )
                st.stop()
        else:
            # For multiple metrics, create a dictionary
            scoring = {
                name: self.scoring_options.get(name) for name in selected_metric_names
            }
            scoring = {k: v for k, v in scoring.items() if v is not None}
            if not scoring:
                st.warning("No valid scoring metrics selected.")
                st.stop()

        # # Get the metric instances based on the selected names
        metrics = [metric_name_to_instance[name] for name in selected_metric_names]

        st.header("Select the hyperparameters that can be used for tuning")
        # Use the appropriate method or attribute to get hyperparameters
        acceptable_ranges = self.select_model_hyperparameters(model)
        X = df[input_feature_names]
        y = df[target_feature_name]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 - split_ratio, random_state=42
        )

        # Perform Grid Search
        st.write("Performing hyperparameter tuning...")

        if isinstance(scoring, dict):
            refit_metric = selected_metric_names[
                0
            ]  # Choose the first metric for refitting
            grid_search = GridSearchCV(
                estimator=model._model,
                param_grid=acceptable_ranges,
                scoring=scoring,
                refit=self.scoring_options.get(refit_metric),
                cv=5,
            )
        else:
            grid_search = GridSearchCV(
                estimator=model._model,
                param_grid=acceptable_ranges,
                scoring=scoring,
                cv=5,
            )

        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_params = grid_search.best_params_
        st.write("Best hyperparameters found:")
        st.write(best_params)

        # Create a new model instance with the best hyperparameters
        tuned_model = model_class(**best_params)

        st.header("You have selected the following pipeline:")
        st.write(f"Dataset: {dataset.name}")
        st.write(f"Target feature: {target_feature.name}")
        st.write(f"Input features: {input_feature_names}")
        st.write(f"Model: {selected_model_name}")
        st.write(f"Metrics: {selected_metric_names}")
        st.write(f"Hyperparameters for tuning: {acceptable_ranges}")
        st.write(f"Dataset split: {split_ratio}")
        pipeline_name = st.text_input("Pipeline name", key="pipeline_name")

        if st.button("Train"):
            # Proceed with training using the tuned model
            pipeline = self.train_pipeline(
                dataset=dataset,
                model=tuned_model,
                input_features=input_features,
                target_feature=target_feature,
                split_ratio=split_ratio,
                metrics_use=metrics,
            )

            st.write("Model trained successfully")
            # Print model
            st.header("Results of training")
            st.write("Predictions:")
            st.write(pipeline._predictions)
            st.write("Metrics results:")
            for metric, result in pipeline._metrics_results:
                st.write(f"- {metric.name}: {result}")

            st.header("Save pipeline")

            pipeline_artifact = pipeline.to_artifact(name=pipeline_name, version="1.0")
            self.automl.registry.register(pipeline_artifact)
            st.write("Pipeline saved successfully")


if __name__ == "__main__":
    app = Modelling()
    app.run()
