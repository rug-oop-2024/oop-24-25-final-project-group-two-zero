
# pipeline.py
from typing import List
import pickle

from autoop.core.ml.model import (
    Model,
    LinearRegressionModel,
    RidgeRegression,
    LinearRegression,
    TreeClassification,
    KNearestNeighbors,
    StoasticGradient
)
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline():
    
    def __init__(self, 
                 metrics: List[Metric],
                 dataset: Dataset, 
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split=0.8,
                 ):
        self._dataset = dataset
        self._model = model # One model used per pipeline
        self._input_features = input_features # Input features must be given
        self._target_feature = target_feature # Target feature must be given
        self._metrics = metrics # Metrics must be given
        self._artifacts = {} # This is added to the thing from inside the pipeline class
        self._split = split # Split must be given, However, there seems to be no problem
                            # If the split is not given, it will default to 0.8
        if target_feature.type == "categorical" and model.type != "classification":
            raise ValueError("Model type must be classification for categorical target feature")
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError("Model type must be regression for continuous target feature")

    def __str__(self):
        """
        Return a string representation of this pipeline.

        Returns
        -------
        str
            A string representation of this pipeline.
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
    def model(self):
        """
        Returns the model used in this pipeline.

        Returns
        -------
        Model
            The model associated with this pipeline.
        """
        return self._model # This returns the model of the pipeline

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Used to get the artifacts generated during the pipeline execution to be saved
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data)))
        artifacts.append(self._model.to_artifact(name=f"pipeline_model_{self._model.type}"))
        return artifacts
    
    def _register_artifact(self, name: str, artifact):
        """
        Registers an artifact with the given name in the pipeline's internal artifact registry.
        
        Parameters
        ----------
        name : str
            The name of the artifact to be registered.
        artifact : dict
            The artifact to be registered. The artifact must have a "type" key that indicates the type of artifact.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """
        Preprocesses the features of the dataset using the artifacts generated during the execution of the pipeline.
        
        This method is used internally by the pipeline to preprocess the features of the dataset before they are used
        to train the model. The artifacts generated by the feature preprocessing are stored in the `_artifacts` dictionary
        using the feature name as the key.
        
        :return: None
        """
        (target_feature_name, target_data, artifact) = preprocess_features([self._target_feature], self._dataset)[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(self._input_features, self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector, sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [data for (feature_name, data, artifact) in input_results]

    def _split_data(self) -> None:
        # Split the data into training and testing sets
        split = self._split
        self._train_X = [vector[:int(split * len(vector))] for vector in self._input_vectors]
        self._test_X = [vector[int(split * len(vector)):] for vector in self._input_vectors]
        self._train_y = self._output_vector[:int(split * len(self._output_vector))]
        self._test_y = self._output_vector[int(split * len(self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Concatenates the input vectors into a single 2D numpy array.

        Parameters
        ----------
        vectors : List[np.array]
            A list of 1D numpy arrays to be concatenated into a single 2D numpy array.

        Returns
        -------
        np.array
            A 2D numpy array where each column corresponds to one of the input vectors.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self):
        # Need to implement this
        X_train = self._compact_vectors(self._train_X)
        Y_train = self._train_y
        self._training_metric_results = []
        self._model.fit(X_train, Y_train)
        for metric in self._metrics:
            result = metric.evaluate(X_train, Y_train)
            self._metrics_results.append((metric, result))
        self._predictions = predictions


    def _evaluate(self):
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def execute(self):
        """
        Executes the pipeline.

        The pipeline is executed in the following order:
        1. Features are preprocessed using the artifacts generated during the execution of the pipeline.
        2. The data is split into training and testing sets.
        3. The model is trained using the training set.
        4. The model is evaluated using the testing set.

        Returns
        -------
        A dictionary containing the evaluation results and the predictions.
        The dictionary has two keys:
        - "metrics": a list of tuples, where each tuple contains the metric name and the evaluation result.
        - "predictions": the predictions made by the model on the testing set.

        This is done
        -------

        ML/pipeline/evaluation: Extend and modify the execute function to return
        the metrics both on the evaluation and training set.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        return {
            "metrics": self._metrics_results,
            "predictions": self._predictions,
        }
        
