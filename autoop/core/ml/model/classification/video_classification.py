from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from ..model import Model
import numpy as np
from typing import Any


class VideoClassificationModel(Model):
    """
    Video classification model using a simple 3D CNN.
    """

    _type = "classification"
    _available_hyperparameters = {
        'input_shape': (16, 64, 64, 3),  # 16 frames of 64x64 RGB images
        'num_classes': 10,
        'optimizer': ['adam', 'sgd', 'rmsprop'],
        'loss': ['categorical_crossentropy', 'sparse_categorical_crossentropy'],
        'metrics': ['accuracy'],
        'epochs': 10,
        'batch_size': 8,
    }
    _supported_feature_types = ['video']
    _supported_target_types = ['categorical']

    def __init__(self, **hyperparameters: Any) -> None:
        """
        Initializes the VideoClassificationModel with hyperparameters.

        Args:
            **hyperparameters: Hyperparameters for the model.
        """
        super().__init__(**hyperparameters)
        self._hyperparameters = {**self._available_hyperparameters, **self._hyperparameters}
        self._build_model()

    def _build_model(self):
        """
        Builds the VideoClassificationModel based on the given hyperparameters.
        """
        input_shape = self._hyperparameters['input_shape']
        num_classes = self._hyperparameters['num_classes']
        optimizer = self._hyperparameters['optimizer']
        loss = self._hyperparameters['loss']
        metrics = self._hyperparameters['metrics']

        self._model = Sequential()
        self._model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
        self._model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        self._model.add(Flatten())
        self._model.add(Dense(128, activation='relu'))
        self._model.add(Dense(num_classes, activation='softmax'))
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the model to the given data.

        Args:
            observations (np.ndarray): The input video data to fit the model to.
            ground_truth (np.ndarray): The target values to fit the model to.
        """
        epochs = self._hyperparameters['epochs']
        batch_size = self._hyperparameters['batch_size']
        loss = self._hyperparameters['loss']

        if 'sparse' in loss:
            pass
        else:
            ground_truth = to_categorical(ground_truth, num_classes=self._hyperparameters['num_classes'])
        self._model.fit(observations, ground_truth, epochs=epochs, batch_size=batch_size)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the labels for the given observations.

        Args:
            observations (np.ndarray): The input video data to predict.

        Returns:
            np.ndarray: The predicted labels.
        """
        predictions = self._model.predict(observations)
        return np.argmax(predictions, axis=1)
