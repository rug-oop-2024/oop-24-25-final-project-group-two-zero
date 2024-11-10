from typing import List, Any
import pickle
import seaborn as sns
import io
import os
import base64
import numpy as np
from autoop.core.ml.model import Model
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
import shap
import streamlit as st
from autoop.core.ml.model.classification import TreeClassification
import pandas as pd
from copy import deepcopy


class Pipeline:
    """A class for executing machine learning pipelines."""

    def __init__(
        self: "Pipeline",
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
            metrics (List[Metric]):
                A list of Metric objects to
                evaluate the model's performance.
            dataset (Dataset): The dataset
                object containing the data for
                training and evaluation.
            model (Model): The model object
                to be trained and evaluated.
            input_features (List[Feature]):
                A list of Feature objects representing
                the input features.
            target_feature (Feature): The Feature object
                representing the target feature.
            split (float, optional): The ratio for splitting
                the dataset into training and evaluation
                sets. Defaults to 0.8.

        Raises:
            ValueError: If the target feature type
                is categorical and the model type is not classification.
            ValueError: If the target feature type
                is numerical and the model type is not regression.
        """
        self._dataset: Dataset = dataset
        self._model: Model = model
        self._input_features: List[Feature] = input_features
        self._target_feature: Feature = target_feature
        self._metrics: List[Metric] = metrics
        self._artifacts: dict = {}
        self._split: float = split
        if target_feature.type == "categorical" and \
                model.type != "classification":
            raise ValueError(
                "Model type must be classification\
                for categorical target feature"
            )
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model type must be regression\
                for continuous target feature"
            )

    @property
    def metadata(self: "Pipeline") -> dict:
        """
        Retrieve the metadata of the pipeline.

        Returns:
            dict: A dictionary containing the following metadata:
                - "parameters": The parameters of the model.
                - "model": The class name of the model.
                - "input_features": A list of names of the input features.
                - "input_feature_types": A list of types of the input features.
                - "target_feature": The name of the target feature.
                - "target_feature_type": The type of the target feature.
                - "split_ratio": The ratio used to split the dataset.
                - "metrics": A list of names of the metrics used.
        """
        return {
            "parameters": self.model.parameters,
            "model": self.model.__class__.__name__,
            "input_features": [
                feature.name for feature in self.input_features
            ],
            "input_feature_types": [
                feature.type for feature in self.input_features
            ],
            "target_feature": self.target_feature.name,
            "target_feature_type": self.target_feature.type,
            "split_ratio": self.split_ratio,
            "metrics": [metric.name for metric in self.metrics],
        }

    @property
    def input_features(self: "Pipeline") -> List[Feature]:
        """
        Return the list Feature objects representing input features.

        Returns:
            List[Feature]: A list of Feature objects
        """
        return self._input_features

    def __str__(self: 'Pipeline') -> str:
        """
        Return a string representation of the Pipeline object.

        The string representation contains
        the model type, input features,
        target feature, split ratio,
        and metrics.

        Returns:
            str: A string representation of
            the Pipeline object.
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
    def target_feature(self: "Pipeline") -> Feature:
        """
        Return the target feature object.

        Returns:
            Feature: The target feature object.
        """
        return self._target_feature

    @property
    def split_ratio(self: "Pipeline") -> float:
        """
        Return the split ratio used to split the dataset.

        Returns:
            float: The split ratio.
        """
        return self._split

    @property
    def metrics(self: "Pipeline") -> List[Metric]:
        """
        Return the metrics used to evaluate the model.

        Returns:
            List[Metric]: A list of Metric objects.
        """
        return deepcopy(self._metrics)

    @property
    def model(self: "Pipeline") -> Model:
        """
        Return the model object used in the pipeline.

        Returns:
            Model: The model object.
        """
        return self._model

    @property
    def artifacts(self: "Pipeline") -> List[Artifact]:
        """
        Return a list of Artifact objects.

        The list of artifacts includes the following:
        - Preprocessing artifacts: Each artifact
        is a dict containing the type of preprocessing
        and the corresponding encoder or scaler.
        - Model artifact: The model artifact is a dict
        containing the model object and its type.

        Returns:
            List[Artifact]: A list of Artifact objects.
        """
        artifacts: List[Artifact] = []
        for name, artifact in self._artifacts.items():
            artifact_type: str = artifact.get("type")
            data: bytes = pickle.dumps(
                artifact["encoder"]
                if artifact_type in ["OneHotEncoder", "LabelEncoder"]
                else artifact["scaler"]
            )
            artifacts.append(Artifact(name=name, data=data))
        pipeline_data: dict = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
        )
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
        )
        return artifacts

    def _register_artifact(
        self: "Pipeline",
        name: str,
        artifact: dict
    ) -> None:
        """
        Register an artifact in the internal dictionary of artifacts.

        Args:
            name (str): The name of the artifact.
            artifact (dict): The artifact
            to be registered.
        """
        self._artifacts[name] = artifact

    def _validate_feature_and_target_types(self: "Pipeline") -> None:
        """
        Validate that the input feature types.

        This method checks that the model supports
        all input feature types and the target feature type.
        If the model does not support a feature type,
        a ValueError is raised.
        raises:
            ValueError: If the model does not support
            any of the input feature types or the target
            feature type.
        """
        feature_types: set = set(
            feature.type for feature in self._input_features
        )
        target_type: str = self._target_feature.type

        # Ensure the model supports all input feature types
        if not all(
            ftype in self._model.supported_feature_types
            for ftype in feature_types
        ):
            raise ValueError(
                f"Model {self._model.__class__.__name__}\
                does not support feature types {feature_types}"
            )

        if target_type not in self._model.supported_target_types:
            raise ValueError(
                f"Model {self._model.__class__.__name__} "
                f"does not support target type {target_type}"
            )

    def _preprocess_features(self: "Pipeline") -> None:
        """
        Preprocess input and target features based
        on their types.

        Preprocesses each feature using the
        `_preprocess_feature_data` method and stores the
        preprocessed data in `_input_vectors` and
        `_output_vector` respectively.

        :return: None
        """
        self._input_vectors: List[np.ndarray] = []
        for feature in self._input_features:
            data = self._dataset.data[feature.name]
            preprocessed_data: np.ndarray = \
                self._preprocess_feature_data(feature, data)
            self._input_vectors.append(preprocessed_data)

        # Preprocess target feature
        target_data: np.ndarray = \
            self._dataset.data[self._target_feature.name]
        self._output_vector = self._preprocess_feature_data(
            self._target_feature, target_data
        )

    def _preprocess_feature_data(
        self: "Pipeline",
        feature: Feature,
        data: pd.Series
    ) -> np.ndarray:
        """
        Preprocess data based on feature type.

        Args:
            feature (Feature): The feature to preprocess.
            data: The data corresponding to the feature.

        Returns:
            np.ndarray: Preprocessed data.
        """
        if feature.type == "image":
            # Assume images are already loaded as arrays
            preprocessed_data = np.stack(data.values)
        elif feature.type == "text":
            preprocessed_data = data.tolist()
        elif feature.type == "audio":
            preprocessed_data = np.stack(data.values)
        elif feature.type == "video":
            preprocessed_data = np.stack(data.values)
        else:
            # For numerical and categorical data,
            # use existing preprocessing
            preprocessed_data = data.values.reshape(-1, 1)
        return preprocessed_data

    def _split_data(self:"Pipeline") -> None:
        """
        Split the preprocessed input and output training and testing sets.

        The data is split based on the specified
        split ratio. The resulting training
        and testing sets are stored in the respective
        attributes.

        Returns:
            None
        """
        split: float = self._split
        self._train_X: List[np.ndarray] = [
            vector[: int(split * len(vector))] for vector in self._input_vectors
        ]
        self._test_X: List[np.ndarray] = [
            vector[int(split * len(vector)) :] for vector in self._input_vectors
        ]
        self._train_y = \
            self._output_vector[: int(split * len(self._output_vector))]
        self._test_y = \
            self._output_vector[int(split * len(self._output_vector)) :]

    def to_artifact(
        self: "Pipeline",
        name: str,
        version: str
    ) -> Artifact:
        """Convert the current pipeline state into an Artifact object."""

        # Define the asset path where the pipeline data will be saved
        asset_dir = "pipelines"
        asset_filename = f"{name}_{version}.pkl"
        asset_path = \
            os.path.normpath(os.path.join(asset_dir, asset_filename))

        # Ensure the directory exists
        os.makedirs(asset_dir, exist_ok=True)

        # Save the pipeline data to the asset path using pickle
        with open(asset_path, "wb") as f:
            pickle.dump(self._dataset.data, f)

        # Prepare metadata with only JSON-serializable data
        pipeline_meta = {
            "parameters": self._model.parameters,
            "model": self._model.__class__.__name__,
            "input_features":
                [feature.name for feature in self._input_features],
            "input_feature_types":
                [feature.type for feature in self._input_features],
            "target_feature": self._target_feature.name,
            "target_feature_type": self._target_feature.type,
            "split": self._split,
            "metrics": [metric.name for metric in self._metrics],
        }

        return Artifact(
            name=name,
            asset_path=asset_path,
            data=None,  # Data is saved externally
            metadata=pipeline_meta,
            version=version,
            type="pipeline",
        )

    def _compact_vectors(
        self: "Pipeline",
        vectors: List
    ) -> np.ndarray | List:
        """
        Compact the input vectors format suitable for the model.

        For models that accept lists (e.g., text data),
        we might need to return the list as is.
        For numerical data, we concatenate the vectors.

        Args:
            vectors (List): The input vectors.

        Returns:
            Any: Compacted data.
        """
        if self._model.supported_feature_types == ["text"]:
            # Return list of strings
            return vectors[0]  # Assuming one text feature
        elif self._model.supported_feature_types == \
                ["image", "audio", "video"]:
            # Stack along the first axis
            return np.concatenate(vectors, axis=0)
        else:
            # For numerical data
            return np.concatenate(vectors, axis=1)

    def _train(self: "Pipeline") -> None:
        """
        Trains the model using the training data.

        This method takes the preprocessed training data
        and fits the model to it.

        Returns:
            None
        """
        X_train: Any = self._compact_vectors(self._train_X)
        Y_train: Any = self._train_y
        # Ensure shapes are correct
        if X_train.shape[0] != len(Y_train):
            raise ValueError("Number of samples in X_train \
                             and Y_train do not match.")
        self._model.fit(X_train, Y_train)

    def _evaluate(self: "Pipeline") -> None:
        """
        Evaluate the trained model using the test data and metrics.

        This method computes predictions on the test data
        and evaluates these predictions
        against the true labels using the specified metrics.
        The results of each metric evaluation
        are stored in the `_metrics_results` attribute, and the
        predictions are stored in the
        `_predictions` attribute.

        Returns:
            None
        """
        X: Any = self._compact_vectors(self._test_X)
        Y: Any = self._test_y
        self._metrics_results: List | Any = []
        predictions: Any = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions: list = predictions

    def _compute_shap_values(self: "Pipeline") -> None:
        """
        Compute SHAP values model to provide explainability insights.

        This method uses SHAP (SHapley Additive exPlanations)
        to compute feature importance values for
        the test data. It selects an appropriate SHAP explainer
        based on the model type, specifically
        using a TreeExplainer for tree-based models
        and a KernelExplainer
        for other model types. In
        case of an exception during the computation, a warning is
        issued, and the SHAP values are set
        to None.

        Raises:
            Exception: If there is an error in
            computing SHAP values,
            it displays a warning with the error message.
        """
        X_test = self._compact_vectors(self._test_X)
        # Choose the appropriate explainer based on the model type
        try:
            if isinstance(self._model._model, (TreeClassification)):
                explainer = shap.TreeExplainer(self._model._model)
            else:
                explainer = shap.KernelExplainer(
                    self._model.predict, X_test[:100]
                )  # Use a subset for performance
            self._shap_values = explainer(X_test)
        except Exception as e:
            st.warning(f"Error computing SHAP values: {e}")
            self._shap_values = None

    def _generate_report(self: "Pipeline") -> None:
        """
        Generate a report containing the results of the evaluation.

        This method generates a report containing the results
        of the evaluation, such as the
        confusion matrix for classification models and the SHAP
        summary plot. The report is
        stored in the `_report` attribute.

        Returns:
            None
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
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_xlabel("Predicted Labels")
            ax_cm.set_ylabel("True Labels")
            plt.title("Confusion Matrix")
            buf_cm = io.BytesIO()
            plt.savefig(buf_cm, format="png")
            buf_cm.seek(0)
            image_png_cm = buf_cm.getvalue()
            buf_cm.close()
            report["confusion_matrix"] = \
                base64.b64encode(image_png_cm).decode("utf-8")
            plt.close(fig_cm)

        # Compute SHAP values
        self._compute_shap_values()

        # Generate SHAP Waterfall Plot
        shap_values = self._shap_values

        # Check if SHAP values are available
        if hasattr(shap_values, "values"):
            # Get feature names
            feature_names = \
                [feature.name for feature in self._input_features]

            # Handle different shapes of shap_values
            if shap_values.values.ndim == 3:
                # Multiclass classification
                num_classes = shap_values.values.shape[1]
                num_classes = num_classes if num_classes > 1 else 2
                class_index = 0  # Default to the first class
                shap_values_sample = shap.Explanation(
                    values=shap_values.values[
                        0, class_index, :
                    ],
                    base_values=shap_values.base_values[
                        0, class_index
                    ],
                    data=shap_values.data[0],
                    feature_names=feature_names,
                )
            elif shap_values.values.ndim == 2:
                shap_values_sample = shap.Explanation(
                    values=shap_values.values[0],
                    base_values=shap_values.base_values[0],
                    data=shap_values.data[0],
                    feature_names=feature_names,
                )
            else:
                st.warning("Unexpected SHAP values shape.")
                return

            max_features = 10

            abs_shap_values = np.abs(shap_values_sample.values)
            sorted_indices = np.argsort(abs_shap_values)[::-1]
            top_indices = sorted_indices[:max_features]

            shap_value_to_plot = shap.Explanation(
                values=shap_values_sample.values[top_indices],
                base_values=shap_values_sample.base_values,
                data=shap_values_sample.data[top_indices],
                feature_names=[
                    shap_values_sample.feature_names[i] for i in top_indices
                ],
            )

            shap.plots.waterfall(shap_value_to_plot, show=False)

            fig = plt.gcf()

            fig.set_size_inches(8, 6)

            # Save the plot to a buffer
            buf_shap = io.BytesIO()
            fig.savefig(buf_shap, format="png", bbox_inches="tight")
            buf_shap.seek(0)
            image_png_shap = buf_shap.getvalue()
            buf_shap.close()

            # Store the plot in the report
            report["shap_summary"] = \
                base64.b64encode(image_png_shap).decode("utf-8")

            # Close the plot to free memory
            plt.close(fig)
        else:
            report["shap_summary"] = None
            st.warning("No SHAP values available\
                        to generate the summary plot.")

        self._report = report

    def _sensitivity_analysis(self: "Pipeline") -> None:
        """
        Perform a sensitivity analysis of the model to each input feature.

        For each feature, it perturbs the feature by
        adding a small value (e.g., 0.1 times the standard deviation)
        and computes the mean absolute change in
        predictions. The results are stored in a dictionary
        where the keys are the feature names and
        the values are the sensitivities.

        Returns:
            None
        """
        X_test = self._compact_vectors(self._test_X)
        sensitivities = {}
        for i, feature in enumerate(self._input_features):
            X_perturbed = X_test.copy()
            perturbation = 0.1 * np.std(X_test[:, i])
            X_perturbed[:, i] += perturbation
            y_pred_original = self._model.predict(X_test)
            y_pred_perturbed = self._model.predict(X_perturbed)
            # Compute the mean absolute change in predictions
            sensitivity = \
                np.mean(np.abs(y_pred_perturbed - y_pred_original))
            sensitivities[feature.name] = sensitivity
        self._sensitivity_results = sensitivities

    def execute(self: "Pipeline") -> dict:
        """
        Executes the pipeline and returns the results.

        The method preprocesses the input data,
        splits it into training and testing sets,
        trains the model, evaluates the model,
        generates a report, and returns the metrics,
        predictions, and report.

        Returns:
            dict: A dictionary containing the
            evaluation metrics, predictions, and report.
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
