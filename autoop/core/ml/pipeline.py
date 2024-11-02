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
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
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
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        (target_feature_name, target_data, artifact) = preprocess_features([self._target_feature], self._dataset)[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(self._input_features, self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        self._output_vector = target_data
        self._input_vectors = [data for (_, data, _) in input_results]

    def _split_data(self) -> None:
        split = self._split
        self._train_X = [vector[: int(split * len(vector))] for vector in self._input_vectors]
        self._test_X = [vector[int(split * len(vector)) :] for vector in self._input_vectors]
        self._train_y = self._output_vector[: int(split * len(self._output_vector))]
        self._test_y = self._output_vector[int(split * len(self._output_vector)) :]

    def to_artifact(self, name: str, version: str) -> Artifact:
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
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        X_train = self._compact_vectors(self._train_X)
        Y_train = self._train_y
        self._model.fit(X_train, Y_train)

    def _evaluate(self) -> None:
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def _generate_report(self) -> None:
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
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        self._generate_report()
        return {
            "metrics": self._metrics_results,
            "predictions": self._predictions,
        }
