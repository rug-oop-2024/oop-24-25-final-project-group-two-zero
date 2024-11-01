# modelling.py
import streamlit as st
import numpy as np
from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import (
            MultipleLinearRegression,
            RidgeRegression,
            LinearRegressionModel
            )
from autoop.core.ml.model.classification import (
            KNearestNeighbors,
            StoasticGradient,
            TreeClassification
            )
from autoop.core.ml.metric import (
            accuracy,
            MeanSquaredError,
            mean_absolute_error,
            r_squared_error,
            specificity,
            F_one_score
            )
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types

# Define the model lists for regression and classification
REGRESSION_MODELS = {
    "MultipleLinearRegression": MultipleLinearRegression,
    "RidgeRegression": RidgeRegression,
    "LinearRegression": LinearRegressionModel
}

CLASSIFICATION_MODELS = {
    "MultipleLinearClassification": StoasticGradient,
    "KNearestNeighbors": KNearestNeighbors,
    "DecisionTreeClassification": TreeClassification
}


# Define the metrics for regression and classification
REGRESSION_METRICS = {
    "Mean Squared Error": MeanSquaredError(),
    "Mean Absolute Error": mean_absolute_error(),
    "R-Squared": r_squared_error(),
}

CLASSIFICATION_METRICS = {
    "Accuracy": accuracy(),
    "F1 Score": F_one_score(),
    "Specificity": specificity(),
}

# Function to get a model instance by name
def get_model(model_name: str, task_type: str) -> Model:
    if task_type == "regression":
        return REGRESSION_MODELS.get(model_name)()
    elif task_type == "classification":
        return CLASSIFICATION_MODELS.get(model_name)()
    else:
        raise ValueError(f"Unknown task type: {task_type}")

# Function to get metrics based on task type
def get_metrics(task_type: str):
    if task_type == "regression":
        return REGRESSION_METRICS
    elif task_type == "classification":
        return CLASSIFICATION_METRICS
    else:
        raise ValueError(f"Unknown task type: {task_type}")

# Initialize AutoML system and Streamlit page
st.set_page_config(page_title="Modelling", page_icon="📈")
st.write("# ⚙ Modelling")
st.write("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()
datasets = automl._registry.list(type="dataset")

if not datasets:
    st.write("No datasets available, please upload a dataset first on the datasets page.")
    st.stop()

# Allow user to select a dataset
selected_dataset = st.selectbox('Choose a dataset:', datasets)
dataset_chosen = automl._registry.get(selected_dataset)

# Get features from the dataset
# Assuming dataset_chosen.features() returns a list of Feature instances
feature_list = detect_feature_types(dataset_chosen)  # Returns a list of Feature instances
feature_names = [feature.name for feature in feature_list]

# Map feature names to Feature instances for easy lookup
feature_dict = {feature.name: feature for feature in feature_list}

# Step 1: Select input features and target feature
input_features_selected_names = st.multiselect("Select input features", feature_names)
target_feature_selected_name = st.selectbox("Select target feature", feature_names)

# Convert selected feature names to Feature instances
input_features_selected = [feature_dict[name] for name in input_features_selected_names]
target_feature_selected = feature_dict[target_feature_selected_name]

# Step 2: Detect task type (classification or regression)
if input_features_selected and target_feature_selected:
    st.write(f"You selected input features: {input_features_selected_names}, and target feature: {target_feature_selected_name}")

    # Determine if the target feature is categorical or numerical
    target_feature_type = target_feature_selected.type

    if target_feature_type == "categorical":
        task_type = "classification"
        st.write("Target feature is categorical, so this is a classification task.")
        available_models = list(CLASSIFICATION_MODELS.keys())
    else:
        task_type = "regression"
        st.write("Target feature is numerical, so this is a regression task.")
        available_models = list(REGRESSION_MODELS.keys())

    # Step 3: Allow the user to select a model based on the task type
    selected_model_name = st.selectbox(f"Select a {task_type} model:", available_models)

    # Step 4: Allow user to select dataset split ratio
    split_ratio = st.slider('Select train-test split ratio', 0.1, 0.9, 0.8)

    # Step 5: Allow user to select metrics based on task type
    available_metrics = get_metrics(task_type)
    selected_metric_names = st.multiselect(f"Select metrics for {task_type}", available_metrics.keys())

    # Convert selected metric names to Metric instances
    selected_metrics = [available_metrics[name] for name in selected_metric_names]

    # Step 6: Display a pipeline summary
# Enhanced Pipeline Summary
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
            model_instance = get_model(selected_model_name, task_type)

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
            for metric, value in results['metrics']:
                st.write(f"{metric.name}: {value}")
        # After reporting the results in modelling.py

        # Step 8: Save the pipeline as an artifact
        st.write("## 💾 Save the Pipeline")
        pipeline_name = st.text_input("Enter a name for the pipeline")
        pipeline_version = st.text_input("Enter a version for the pipeline", "1.0.0")

        if st.button("Save Pipeline"):
            if pipeline_name:
                # Convert the pipeline into an artifact
                pipeline_artifact = pipeline.to_artifact(name=pipeline_name, version=pipeline_version)
                # Register the pipeline artifact
                automl.registry.register(pipeline_artifact)
                st.success(f"Pipeline '{pipeline_name}' version '{pipeline_version}' has been saved.")
            else:
                st.warning("Please enter a name for the pipeline.")


    else:
        st.write("Please select a model and metrics to proceed.")