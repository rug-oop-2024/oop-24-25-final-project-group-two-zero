from typing import List
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import numpy as np
from autoop.core.ml.model import Model
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import shap
from typing import List, Dict, Any
import streamlit as st
from autoop.core.ml.model.classification import TreeClassification

class Pipeline:
    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: Model,
        input_features: List[Feature],
        target_feature: Feature,
        split: float = 0.8,
    ) -> None:
        """
        Initialize the Pipeline object.

        Args:
            metrics (List[Metric]): A list of Metric objects to evaluate the model's performance.
            dataset (Dataset): The dataset object containing the data for training and evaluation.
            model (Model): The model object to be trained and evaluated.
            input_features (List[Feature]): A list of Feature objects representing the input features.
            target_feature (Feature): The Feature object representing the target feature.
            split (float, optional): The ratio for splitting the dataset into training and evaluation sets. Defaults to 0.8.

        Raises:
            ValueError: If the target feature type is categorical and the model type is not classification.
            ValueError: If the target feature type is numerical and the model type is not regression.
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if target_feature.type == "categorical" and model.type != "classification":
            raise ValueError("Model type must be classification for categorical target feature")
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError("Model type must be regression for continuous target feature")

    def __str__(self) -> str:
        """
        Return a string representation of the Pipeline object.

        The string representation contains the model type, input features, target feature, split ratio, and metrics.

        Returns:
            str: A string representation of the Pipeline object.
        """
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """
        Return the model object used in the pipeline.

        Returns:
            Model: The model object.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Return a list of Artifact objects containing the preprocessing artifacts and the model artifact.

        The list of artifacts includes the following:
        - Preprocessing artifacts: Each artifact is a dict containing the type of preprocessing and the corresponding encoder or scaler.
        - Model artifact: The model artifact is a dict containing the model object and its type.

        Returns:
            List[Artifact]: A list of Artifact objects.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            data = pickle.dumps(artifact["encoder"] if artifact_type in ["OneHotEncoder", "LabelEncoder"] else artifact["scaler"])
            artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data)))
        artifacts.append(self._model.to_artifact(name=f"pipeline_model_{self._model.type}"))
        return artifacts

    def _register_artifact(self, name: str, artifact: dict) -> None:
        """
        Registers an artifact in the internal dictionary of artifacts.

        Args:
            name (str): The name of the artifact.
            artifact (dict): The artifact to be registered.
        """
        self._artifacts[name] = artifact

    def _validate_feature_and_target_types(self):
        """
        Validates that the model supports the selected feature and target types.
        """
        feature_types = set(feature.type for feature in self._input_features)
        target_type = self._target_feature.type

        # Ensure the model supports all input feature types
        if not all(ftype in self._model.supported_feature_types for ftype in feature_types):
            raise ValueError(f"Model {self._model.__class__.__name__} does not support feature types {feature_types}")

        if target_type not in self._model.supported_target_types:
            raise ValueError(f"Model {self._model.__class__.__name__} does not support target type {target_type}")

    def _preprocess_features(self) -> None:
        """
        Preprocesses the input features and target feature.
        """
        # For each feature, apply preprocessing based on its type
        self._input_vectors = []
        for feature in self._input_features:
            data = self._dataset.data[feature.name]
            preprocessed_data = self._preprocess_feature_data(feature, data)
            self._input_vectors.append(preprocessed_data)

        # Preprocess target feature
        target_data = self._dataset.data[self._target_feature.name]
        self._output_vector = self._preprocess_feature_data(self._target_feature, target_data)

    def _preprocess_feature_data(self, feature: Feature, data):
        """
        Preprocesses data based on feature type.

        Args:
            feature (Feature): The feature to preprocess.
            data: The data corresponding to the feature.

        Returns:
            np.ndarray: Preprocessed data.
        """
        if feature.type == 'image':
            # Assume images are already loaded as arrays
            preprocessed_data = np.stack(data.values)
        elif feature.type == 'text':
            preprocessed_data = data.tolist()
        elif feature.type == 'audio':
            preprocessed_data = np.stack(data.values)
        elif feature.type == 'video':
            preprocessed_data = np.stack(data.values)
        else:
            # For numerical and categorical data, use existing preprocessing
            preprocessed_data = data.values.reshape(-1, 1)
        return preprocessed_data

    def _split_data(self) -> None:
        """
        Splits the preprocessed input and output data into training and testing sets.

        The data is split based on the specified split ratio. The resulting training
        and testing sets are stored in the respective attributes.

        Returns:
            None
        """
        split = self._split
        self._train_X = [vector[: int(split * len(vector))] for vector in self._input_vectors]
        self._test_X = [vector[int(split * len(vector)) :] for vector in self._input_vectors]
        self._train_y = self._output_vector[: int(split * len(self._output_vector))]
        self._test_y = self._output_vector[int(split * len(self._output_vector)) :]

    def to_artifact(self, name: str, version: str) -> Artifact:
        """
        Converts the Pipeline object to an Artifact.

        The Artifact contains the model, input features, target feature, split ratio, metrics, and
        preprocessing artifacts.

        Args:
            name (str): The name of the pipeline.
            version (str): The version of the pipeline.

        Returns:
            Artifact: An Artifact object containing the pipeline's data.
        """
        pipeline_data = {
            'model': self._model,
            'input_features': self._input_features,
            'target_feature': self._target_feature,
            'split': self._split,
            'metrics': self._metrics,
            'preprocessing_artifacts': self._artifacts,
        }
        data_bytes = pickle.dumps(pipeline_data)
        asset_path = os.path.normpath(os.path.join("pipelines", f"{name}_{version}.pkl"))
        return Artifact(
            name=name,
            asset_path=asset_path,
            data=data_bytes,
            version=version,
            type='pipeline',
        )

    def _compact_vectors(self, vectors: List) -> Any:
        """
        Compacts the input vectors into a format suitable for the model.

        For models that accept lists (e.g., text data), we might need to return the list as is.
        For numerical data, we concatenate the vectors.

        Args:
            vectors (List): The input vectors.

        Returns:
            Any: Compacted data.
        """
        if self._model.supported_feature_types == ['text']:
            # Return list of strings
            return vectors[0]  # Assuming one text feature
        elif self._model.supported_feature_types == ['image', 'audio', 'video']:
            # Stack along the first axis
            return np.concatenate(vectors, axis=0)
        else:
            # For numerical data
            return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """
        Trains the model using the training data.

        This method trains the model using the training data, which is
        stored in the `_train_X` and `_train_y` attributes. The input
        vectors are concatenated into a single array using the
        `_compact_vectors` method, and then the model is fit to the
        resulting array and the labels.

        The trained model is stored in the `_model` attribute.
        """
        X_train = self._compact_vectors(self._train_X)
        Y_train = self._train_y
        self._model.fit(X_train, Y_train)

    def _evaluate(self) -> None:
        """
        Evaluates the trained model using the test data and metrics.

        This method computes predictions on the test data and evaluates these predictions
        against the true labels using the specified metrics. The results of each metric evaluation
        are stored in the `_metrics_results` attribute, and the predictions are stored in the
        `_predictions` attribute.

        Returns:
            None
        """
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def _compute_shap_values(self):
        """
        Computes SHAP values for the test data.

        This method uses the trained model to compute SHAP values for the test dataset.
        The SHAP values are stored in the `_shap_values` attribute.
        """
        X_test = self._compact_vectors(self._test_X)
        # Choose the appropriate explainer based on the model type
        try:
            if isinstance(self._model._model, (TreeClassification)):
                explainer = shap.TreeExplainer(self._model._model)
            else:
                explainer = shap.KernelExplainer(self._model.predict, X_test[:100])  # Use a subset for performance
            self._shap_values = explainer(X_test)
        except Exception as e:
            st.warning(f"Error computing SHAP values: {e}")
            self._shap_values = None



    def _generate_report(self):
        """
        Generates a report containing explainability plots.

        The report includes a confusion matrix for classification models and SHAP plots
        for both classification and regression models.
        """
        report = {}
        import matplotlib.pyplot as plt

        # Generate Confusion Matrix for Classification Models
        if self._model.type == "classification":
            from sklearn.metrics import confusion_matrix
            y_true = self._test_y
            y_pred = self._predictions
            cm = confusion_matrix(y_true, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel('Predicted Labels')
            ax_cm.set_ylabel('True Labels')
            plt.title('Confusion Matrix')
            buf_cm = io.BytesIO()
            plt.savefig(buf_cm, format='png')
            buf_cm.seek(0)
            image_png_cm = buf_cm.getvalue()
            buf_cm.close()
            report['confusion_matrix'] = base64.b64encode(image_png_cm).decode('utf-8')
            plt.close(fig_cm)

        # Compute SHAP values
        self._compute_shap_values()

        # Generate SHAP Waterfall Plot
        shap_values = self._shap_values

        # Check if SHAP values are available
        if hasattr(shap_values, 'values'):
            # Get feature names
            feature_names = [feature.name for feature in self._input_features]

            # Handle different shapes of shap_values
            if shap_values.values.ndim == 3:
                # Multiclass classification
                num_classes = shap_values.values.shape[1]
                # Ensure class_index is within bounds
                class_index = 0  # Default to the first class
                # Optionally, you can let the user select the class index
                shap_value_to_plot = shap.Explanation(
                    values=shap_values.values[0, class_index, :],
                    base_values=shap_values.base_values[0, class_index],
                    data=shap_values.data[0],
                    feature_names=feature_names
                )
            elif shap_values.values.ndim == 2:
                # Binary classification or regression
                shap_value_to_plot = shap.Explanation(
                    values=shap_values.values[0],
                    base_values=shap_values.base_values[0],
                    data=shap_values.data[0],
                    feature_names=feature_names
                )
            else:
                st.warning("Unexpected SHAP values shape.")
                return

            # Create a figure to hold the plot
            plt.figure()
            # Generate the waterfall plot without displaying it
            shap.plots.waterfall(shap_value_to_plot, show=False)
            # Save the plot to a buffer
            buf_shap = io.BytesIO()
            plt.savefig(buf_shap, format='png', bbox_inches='tight')
            buf_shap.seek(0)
            image_png_shap = buf_shap.getvalue()
            buf_shap.close()
            # Store the plot in the report
            report['shap_summary'] = base64.b64encode(image_png_shap).decode('utf-8')
            # Close the plot to free memory
            plt.close()
        else:
            report['shap_summary'] = None
            st.warning("No SHAP values available to generate the summary plot.")

        # Assign the report to the instance variable
        self._report = report



    def _sensitivity_analysis(self):
        """
        Performs sensitivity analysis on the test data.

        For each feature, this method perturbs the feature values and observes the change in predictions.
        The results are stored in the `_sensitivity_results` attribute.
        """
        X_test = self._compact_vectors(self._test_X)
        sensitivities = {}
        for i, feature in enumerate(self._input_features):
            X_perturbed = X_test.copy()
            # Perturb the feature by adding a small value (e.g., 0.1 times the standard deviation)
            perturbation = 0.1 * np.std(X_test[:, i])
            X_perturbed[:, i] += perturbation
            y_pred_original = self._model.predict(X_test)
            y_pred_perturbed = self._model.predict(X_perturbed)
            # Compute the mean absolute change in predictions
            sensitivity = np.mean(np.abs(y_pred_perturbed - y_pred_original))
            sensitivities[feature.name] = sensitivity
        self._sensitivity_results = sensitivities

    def execute(self) -> dict:
        """
        Executes the pipeline and returns the results.

        The method preprocesses the input data, splits it into training and testing sets,
        trains the model, evaluates the model, generates a report, and returns the metrics,
        predictions, and report.

        Returns:
            dict: A dictionary containing the evaluation metrics, predictions, and report.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        self._generate_report()
        return {
            "metrics": self._metrics_results,
            "predictions": self._predictions,
            "report": self._report,
        }
