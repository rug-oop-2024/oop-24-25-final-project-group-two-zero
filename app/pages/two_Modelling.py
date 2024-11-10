import streamlit as st
from typing import List
import pandas as pd
import pickle
import os
import numpy as np
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
from autoop.core.ml.metric import Metric

st.set_page_config(page_title="Modelling", page_icon="📈")

"""This module provides a Streamlit interface for training a pipeline."""


st.write("# ⚙ Modelling")


class Modelling:
    """This class is responsible for training a pipeline."""

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

    def __init__(self: "Modelling") -> None:
        """
        Initialize the Modelling class.

        This method initializes the AutoML system and an empty list to
        store the available datasets.

        :return: None
        """
        self.automl = AutoMLSystem.get_instance()
        self.datasets = self.automl.registry.list(type="dataset")

    def train_pipeline(
        self: "Modelling",
        dataset: Dataset,
        model: Model,
        input_features: List[Feature],
        target_feature: Feature,
        split_ratio: float,
        metrics_use: List["Metric"],
    ) -> Pipeline:
        """
        Train a pipeline.

        Args:
            dataset (Dataset): The dataset to train the pipeline on.
            model (Model): The model to use in the pipeline.
            input_features (List[Feature]): The input features for the model.
            target_feature (Feature): The target feature for the model.
            split_ratio (float): The ratio to split the dataset.
            metrics_use (List[Metric]): The metrics to use for evaluation.
        
        Returns:
            Pipeline: The trained pipeline.
        """
        pipeline = Pipeline(
            metrics=metrics_use,
            dataset=dataset,
            model=model,
            input_features=input_features,  # Ensure this is correctly provided
            target_feature=target_feature,
            split=split_ratio,
        )
        values = pipeline.execute()
        print(values["predictions"])
        return pipeline

    def select_model_hyperparameters(
        self: "Modelling",
        model_instance: Model
    ) -> dict:
        """
        Allow the user to specify acceptable ranges
        or options for all hyperparameters.

        Args:
            model_instance (Model): The model instance.

        Returns:
            dict: A dictionary containing acceptable
                ranges for hyperparameters.
        """
        st.write("### Model Hyperparameters")
        hyperparameters = model_instance.available_hyperparameters
        print("Available Hyperparameters:", hyperparameters)

        acceptable_ranges = {}
        missing_selections = False

        for param, default in hyperparameters.items():
            print("Processing Hyperparameter:", param, default)

            if isinstance(default, bool):
                # Boolean hyperparameters
                acceptable_values = st.multiselect(
                    f"Select acceptable value(s) for '{param}':",
                    options=[True, False],
                    default=[default],
                )
                acceptable_ranges[param] = (
                    acceptable_values if acceptable_values else [default]
                )
            elif isinstance(default, list):
                # Categorical hyperparameters
                # Assuming 'default' is a list of possible options
                acceptable_values = st.multiselect(
                    f"Select acceptable value(s) for '{param}':",
                    options=default,
                    default=default,
                )
                acceptable_ranges[param] = (
                    acceptable_values if acceptable_values else default
                )
            elif isinstance(default, int):
                # Integer hyperparameters with a range
                st.write(f"**Define the range for int hyperparameter '{param}':**")
                min_value = st.number_input(
                    f"Minimum acceptable value for '{param}'",
                    value=max(1, default - 10),
                    step=1,
                )
                max_value = st.number_input(
                    f"Maximum acceptable value for '{param}'",
                    value=default + 10,
                    step=1,
                )
                num_samples = st.slider(
                    f"Number of samples for '{param}'",
                    min_value=2,
                    max_value=20,
                    value=5,
                )
                # Generate a list of integer values using linspace
                linspace_values = np.linspace(
                    min_value,
                    max_value,
                    num=num_samples
                )
                int_values = [int(round(val)) for val in linspace_values]
                # Remove duplicates and sort
                unique_int_values = sorted(set(int_values))
                acceptable_ranges[param] = unique_int_values

                # Display the generated options for clarity
                st.write(
                    f"""**Generated options
                      for '{param}':** {unique_int_values}"""
                )
            elif isinstance(default, float):
                # Float hyperparameters with a range
                st.write(
                    f"""**Define the
                    range for float hyperparameter '{param}':**"""
                )
                min_value = st.number_input(
                    f"Minimum acceptable value for '{param}'",
                    value=default,
                    format="%.4f",
                )
                max_value = st.number_input(
                    f"Maximum acceptable value for '{param}'",
                    value=default + 1.0,
                    format="%.4f",
                )
                num_samples = st.slider(
                    f"Number of samples for '{param}'",
                    min_value=2,
                    max_value=20,
                    value=5,
                )
                # Generate a list of float values using linspace
                linspace_values = np.linspace(
                    min_value, max_value, num=num_samples
                )
                float_values = linspace_values.tolist()
                acceptable_ranges[param] = float_values

                # Display the generated options for clarity
                st.write(
                    f"""**Generated options for '{param}':** {float_values}"""
                )
            elif default is None:
                # Hyperparameters that can accept None or specific types
                st.write(
                    f"**Define acceptable value(s) for '{param}':**"
                )
                predefined_options = ["None"]

                # Determine the type based on model's hyperparameter definition
                if hasattr(model_instance, "hyperparameter_type"):
                    hyperparam_type = \
                        model_instance.hyperparameter_type(param)
                    if hyperparam_type == "int":
                        predefined_options += [1, 2, 3, 4, 5]
                    elif hyperparam_type == "float":
                        predefined_options += \
                            [0.1, 0.5, 1.0, 1.5, 2.0]
                    elif hyperparam_type == "str":
                        predefined_options += \
                            ["option1", "option2", "option3"]
                else:
                    predefined_options += [1, 2, 3, 4, 5]

                acceptable_values = st.multiselect(
                    f"Select acceptable options for '{param}':",
                    options=predefined_options,
                    default=["None"],
                )

                # Convert 'None' string to actual None type
                processed_values = []
                for val in acceptable_values:
                    if val == "None":
                        processed_values.append(None)
                    else:
                        processed_values.append(val)
                acceptable_ranges[param] = (
                    processed_values if \
                        processed_values else [None]
                )
            elif isinstance(default, str):
                # String hyperparameters with predefined options
                predefined_options = [
                    default,
                    "option1",
                    "option2",
                    "option3",
                ]  # Example options
                acceptable_values = st.multiselect(
                    f"Select acceptable value(s) for '{param}':",
                    options=predefined_options,
                    default=[default],
                )
                acceptable_ranges[param] = (
                    acceptable_values \
                        if acceptable_values else [default]
                )
            else:
                predefined_options = [
                    str(default),
                    "option1",
                    "option2",
                    "option3",
                ]  # Example options
                acceptable_values = st.multiselect(
                    f"Select acceptable value(s) for '{param}':",
                    options=predefined_options,
                    default=[str(default)],
                )
                acceptable_ranges[param] = (
                    acceptable_values \
                        if acceptable_values else [str(default)]
                )

            # Check if any hyperparameter has no selected values
            if not acceptable_ranges[param]:
                st.warning(
                    f"""No acceptable values \
                        defined for '{param}'. Please make a selection."""
                )
                missing_selections = True

        if missing_selections:
            st.warning(
                """Please select acceptable \
                    values for all hyperparameters before proceeding."""
            )
            st.stop()
        else:
            st.success("All hyperparameters \
                       have acceptable values selected.")
            return acceptable_ranges

    def run(self: "Modelling") -> None:
        """
        This function is the entrypoint for the modeling page.

        It is responsible for displaying the dataset
        features, selecting the target feature,
        selecting the model, selecting the metrics,
        selecting hyperparameters for tuning,
        performing hyperparameter tuning, and saving the pipeline.
        returns: None
        """
        dataset_options = {
            f"{dataset.name}": dataset.id for dataset in self.datasets
        }

        selected_dataset_name = st.selectbox(
            "Select a dataset", list(dataset_options.keys())
        )
        dataset_id = dataset_options[selected_dataset_name]
        dataset_artifact = self.automl.registry.get(dataset_id)
        dataset = Dataset.from_artifact(dataset_artifact)

        st.write(f"Selected dataset: {dataset.name}")
        df = dataset.to_dataframe()
        features = detect_feature_types(dataset)

        for feature in features:
            st.write(f"- {feature.name}: {feature.type}")

        st.header("Select the split ratio")
        split_ratio = st.slider(
            "Select the split ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.8
        )

        st.header("Select the target feature")
        target_feature = st.selectbox(
            "Select the target feature",
            [feature for feature in features]
        )
        input_features = st.multiselect(
            "Select the input features",
            [feature for feature in features \
             if feature != target_feature],
        )
        if not input_features:
            st.warning("Please select at least one input feature.")
            st.stop()

        input_feature_names = [
            feature.name for feature in input_features
        ]
        target_feature_name = target_feature.name

        st.header("Select the model")
        model_options = (
            self.CLASSIFICATION_MODELS
            if target_feature.type == "categorical"
            else self.REGRESSION_MODELS
        )
        model_name_to_class = {
            name: cls for name, cls in model_options.items()
        }
        selected_model_name = st.selectbox(
            "Select the model",
            list(model_name_to_class.keys())
        )
        model_class = model_name_to_class[selected_model_name]
        model = model_class()

        st.header("Select the metrics")
        metric_options = (
            self.CLASSIFICATION_METRICS
            if target_feature.type == "categorical"
            else self.REGRESSION_METRICS
        )
        metric_name_to_instance = {
            name: instance for name, instance in metric_options.items()
        }
        selected_metric_names = st.multiselect(
            "Select the metrics",
            list(metric_name_to_instance.keys())
        )
        if not selected_metric_names:
            st.warning("Please select at least one metric.")
            st.stop()

        scoring = (
            self.scoring_options[selected_metric_names[0]]
            if len(selected_metric_names) == 1
            else {
                name: self.scoring_options[name]
                for name in selected_metric_names
                if name in self.scoring_options
            }
        )

        metrics = [
            metric_name_to_instance[name] \
                for name in selected_metric_names
        ]
        st.header("Select hyperparameters for tuning")
        acceptable_ranges = self.select_model_hyperparameters(model)

        X = df[input_feature_names]
        y = df[target_feature_name]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 - split_ratio, random_state=42
        )

        st.write("Performing hyperparameter tuning...")
        param_grid = {
            k: acceptable_ranges.get(k, [v]) for k, v in acceptable_ranges.items()
        }

        refit_metric = (
            selected_metric_names[0] if isinstance(scoring, dict) else scoring
        )
        grid_search = GridSearchCV(
            estimator=model._model,
            param_grid=param_grid,
            scoring=scoring,
            refit=(
                self.scoring_options[refit_metric]
                if isinstance(scoring, dict)
                else refit_metric
            ),
            cv=5,
        )
        # Add a spinner to indicate progress
        with st.spinner("Running Grid Search..."):
            try:
                grid_search.fit(X_train, y_train)
                best_params = grid_search.best_params_
            except Exception as e:
                st.error(f"An error occurred during Grid Search: {e}")
                st.stop()
        st.write("Best hyperparameters found:", best_params)
        print(best_params)
        tuned_model = model_class(**best_params)

        st.header("You have selected the following pipeline:")
        st.write(f"Dataset: {dataset.name}")
        st.write(f"Target feature: {target_feature.name}")
        st.write(f"Input features: {input_feature_names}")
        st.write(f"Model: {selected_model_name}")
        st.write(f"Metrics: {selected_metric_names}")
        st.write(f"Hyperparameters for tuning: {best_params}")
        st.write(f"Dataset split: {split_ratio}")

        pipeline_name = st.text_input("Pipeline name", key="pipeline_name")
        if st.button("Train"):
            pipeline = self.train_pipeline(
                dataset=dataset,
                model=tuned_model,
                input_features=input_features,
                target_feature=target_feature,
                split_ratio=split_ratio,
                metrics_use=metrics,
            )

            st.success("Pipeline trained successfully!")

            if pipeline_name:
                # Define the directory and file path
                pipeline_dir = "saved_pipelines"
                os.makedirs(pipeline_dir, exist_ok=True)
                pipeline_path = os.path.join(pipeline_dir, f"{pipeline_name}.pkl")

                # Save the pipeline using pickle
                with open(pipeline_path, "wb") as f:
                    pickle.dump(pipeline, f)

                st.success(f"Pipeline saved successfully as '{pipeline_name}.pkl'!")


if __name__ == "__main__":
    app = Modelling()
    app.run()
