# deployment.py
import streamlit as st
import pandas as pd
import pickle
from typing import List
from autoop.core.ml.feature import Feature
from app.core.system import AutoMLSystem
from autoop.core.ml.artifact import Artifact

# Add this function in deployment.py
import numpy as np

def preprocess_new_data(df: pd.DataFrame, input_features: List[Feature], preprocessing_artifacts: dict) -> np.ndarray:
    """
    Preprocess new data using the saved preprocessing artifacts.

    Parameters:
    - df: The new data as a pandas DataFrame.
    - input_features: List of Feature objects representing the input features.
    - preprocessing_artifacts: Dictionary of preprocessing artifacts saved in the pipeline.

    Returns:
    - X_new: The preprocessed feature matrix as a numpy array.
    """
    preprocessed_vectors = []
    for feature in input_features:
        feature_name = feature.name
        if feature_name not in df.columns:
            raise ValueError(f"Feature '{feature_name}' is missing in the uploaded data.")

        if feature.type == "categorical":
            # Use the saved OneHotEncoder
            artifact = preprocessing_artifacts[feature_name]
            encoder = pickle.loads(artifact['encoder'])
            feature_data = df[feature_name].values.reshape(-1, 1)
            transformed_data = encoder.transform(feature_data).toarray()
        elif feature.type == "numerical":
            # Use the saved StandardScaler
            artifact = preprocessing_artifacts[feature_name]
            scaler = pickle.loads(artifact['scaler'])
            feature_data = df[feature_name].values.reshape(-1, 1)
            transformed_data = scaler.transform(feature_data)
        else:
            raise ValueError(f"Unknown feature type: {feature.type}")
        preprocessed_vectors.append(transformed_data)
    # Concatenate the preprocessed feature vectors
    X_new = np.concatenate(preprocessed_vectors, axis=1)
    return X_new


st.set_page_config(page_title="Deployment", page_icon="🚀")
st.write("# 🚀 Deployment")
st.write("In this section, you can load saved pipelines and perform predictions.")

# Initialize AutoML system
automl = AutoMLSystem.get_instance()

# Get list of saved pipelines
pipelines = automl.registry.list(type="pipeline")
if not pipelines:
    st.write("No saved pipelines available.")
    st.stop()

# Display the list of pipelines
st.write("## Available Pipelines")
pipeline_names = [f"{pipeline.name} (Version: {pipeline.version})" for pipeline in pipelines]
selected_pipeline_name = st.selectbox("Select a pipeline:", pipeline_names)

# Find the selected pipeline
selected_index = pipeline_names.index(selected_pipeline_name)
selected_pipeline = pipelines[selected_index]

# Load the pipeline data
pipeline_data = pickle.loads(selected_pipeline.data)

# Display the pipeline summary
st.write("## 📋 Pipeline Summary")
st.markdown(f"""
- **Name**: `{selected_pipeline.name}`
- **Version**: `{selected_pipeline.version}`
- **Model**: `{pipeline_data['model'].__class__.__name__}`
- **Input Features**: `{', '.join([f.name for f in pipeline_data['input_features']])}`
- **Target Feature**: `{pipeline_data['target_feature'].name}`
- **Metrics**: `{', '.join([m.name for m in pipeline_data['metrics']])}`
- **Train-Test Split Ratio**: `{pipeline_data['split']}`
""")


# Allow the user to upload a CSV file for predictions
st.write("## 📁 Upload Data for Prediction")
uploaded_file = st.file_uploader("Choose a CSV file for predictions", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df_new = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data")
    st.dataframe(df_new)

    # Preprocess the data using the pipeline's preprocessing artifacts
    try:
        X_new = preprocess_new_data(df_new, pipeline_data['input_features'], pipeline_data['preprocessing_artifacts'])
        # Make predictions
        model = pipeline_data['model']
        predictions = model.predict(X_new)
        st.write("## 🔮 Predictions")
        st.write(predictions)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.write("Please upload a CSV file to make predictions.")

