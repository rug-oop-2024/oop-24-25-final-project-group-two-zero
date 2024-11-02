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
        if target_feature.type == "numerical" and model.type != "regression":
            raise ValueError("Model type must be regression for numerical target feature")

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

    def _preprocess_features(self) -> None:
        """
        Preprocesses the input features and target feature using the preprocess_features function.

        The function takes the list of input features and the target feature, and preprocesses them
        using the preprocess_features function. The preprocessed data is stored in the _input_vectors
        and _output_vector attributes of the Pipeline object.

        Returns:
            None
        """
        (target_feature_name, target_data, artifact) = preprocess_features([self._target_feature], self._dataset)[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(self._input_features, self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        self._output_vector = target_data
        self._input_vectors = [data for (_, data, _) in input_results]

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

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Concatenates the input vectors into a single array along the second axis.

        The input vectors are concatenated into a single array, which is
        returned as the output. The concatenation is done along the second
        axis (i.e., axis=1).

        Args:
            vectors (List[np.array]): The input vectors to be concatenated.

        Returns:
            np.array: The concatenated array.
        """
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

    def _generate_report(self) -> None:
        """
        Generates a report containing a confusion matrix for classification models.

        The report is generated only for classification models, and it contains a
        confusion matrix as a base64-encoded PNG image.

        Returns:
            None
        """
        report = {}
        if self._model.type == "classification":
            from sklearn.metrics import confusion_matrix
            y_true = self._test_y
            y_pred = self._predictions
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')
            plt.title('Confusion Matrix')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_png = buf.getvalue()
            buf.close()
            report['confusion_matrix'] = base64.b64encode(image_png).decode('utf-8')
        self._report = report

    def execute(self) -> dict:
        """
        Executes the pipeline and returns the results.

        The method preprocesses the input data, splits it into training and testing sets, trains the model, evaluates the model, and generates a report.

        Returns:
            A dictionary containing the evaluation metrics and the predictions of the model.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        self._generate_report()
        return {
            "metrics": self._metrics_results,
            "predictions": self._predictions,
        }
