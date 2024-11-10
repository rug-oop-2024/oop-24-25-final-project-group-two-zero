import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import get_metric

class deployment:
    def __init__(self):
        pass
    def run(self):
        # Define the directory where pipelines are saved
        pipeline_dir = "saved_pipelines"

        st.set_page_config(page_title="Deployment", page_icon="🚀")

        st.write("# 🚀 Deployment")

        if not os.path.exists(pipeline_dir):
            st.write("No pipelines available.")
        else:
            pipeline_files = [f for f in os.listdir(pipeline_dir) if f.endswith(".pkl")]
            if not pipeline_files:
                st.write("No pipelines available.")
            else:
                st.header("Select a pipeline")
                selected_pipeline_file = st.selectbox("Select a pipeline", pipeline_files)
                pipeline_path = os.path.join(pipeline_dir, selected_pipeline_file)
                
                # Load the entire pipeline object
                try:
                    with open(pipeline_path, 'rb') as f:
                        pipeline = pickle.load(f)
                    st.success(f"Pipeline '{selected_pipeline_file}' loaded successfully!")
                except Exception as e:
                    st.error(f"Failed to load pipeline: {e}")
                    st.stop()
                
                st.write(f"## Selected Pipeline: {selected_pipeline_file}")
                
                # Display pipeline metadata
                st.write("### Pipeline Metadata")
                for key, value in pipeline.metadata.items():
                    st.write(f"- **{key}**: {value}")
                
                st.write("## Upload a Dataset for Prediction")
                uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

                if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file)

                    st.write("### Input Data")
                    st.write(df.head())

                    # Prepare the dataset
                    dataset = Dataset.from_dataframe(
                        df,
                        name="InputDataset",
                        asset_path="",
                        version="1.0",
                    )
                    pipeline.dataset = dataset

                    # Prepare observations
                    input_feature_names = [feature.name for feature in pipeline.input_features]  # Adjust based on implementation
                    observations = df[input_feature_names].values

                    # Make predictions
                    predictions = pipeline.model.predict(observations)

                    st.write("## Predictions")
                    st.write(predictions)

                    # Evaluate metrics if ground truth is available
                    if pipeline.target_feature.name in df.columns:
                        ground_truth = df[pipeline.target_feature.name].values
                        st.write("## Evaluation Metrics")
                        for metric in pipeline.metrics:
                            score = metric.evaluate(predictions, ground_truth)
                            st.write(f"- **{metric.name}**: {score}")
                    else:
                        st.write("Ground truth not found in dataset. Skipping metric evaluation.")

if __name__ == "__main__":
    deploy = deployment()
    deploy.run()