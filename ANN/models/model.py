"""Generic Model Interface"""
import warnings
from abc import ABC, abstractmethod
from random import shuffle as list_shuffle
from typing import Tuple

import cupy as cp
import numpy as np
from tqdm import tqdm

from ANN.errors import ShapeError
from ANN.layers import Layer
from ANN.optimizers.optimizer import Optimizer


class Model(ABC):
    """Generic Model Interface"""

    def __init__(
        self,
        layers: list[Layer],
        optimizer: Optimizer,
    ):
        self.optimizer = optimizer
        self.layers = layers
        self.history = {"training_loss": {}, "validation_loss": {}}
        self.initialized = False

    def train(
        self,
        train_x: np.typing.NDArray[np.float32],
        train_y: np.typing.NDArray[np.float32],
        epochs: int,
        batch_size: int,
        val_x: np.typing.NDArray[np.float32] = None,
        val_y: np.typing.NDArray[np.float32] = None,
        shuffle: bool = True,
    ):
        """Method to train network on provided dataset."""
        if train_x.shape[0] != train_y.shape[0]:
            raise ShapeError(
                f"Mismatched number of samples in training inputs and targets:"
                f" {train_x.shape[0]} != {train_y.shape[0]}"
            )
        set_list = [(train_x, "training")]
        if not val_x is None and not val_y is None:
            if val_x.shape[0] != val_y.shape[0]:
                raise ShapeError(
                    f"Mismatched number of samples in validation inputs and targets:"
                    f" {val_x.shape[0]} != {val_y.shape[0]}"
                )
            set_list.append((val_x, "validation"))

        for input_x, set_name in set_list:
            if len(input_x.shape) < 2:
                raise ShapeError(
                    f"Model expected {set_name} input vector of shape (n_samples, (input_shape)))"
                    f" received {input_x.shape} instead."
                )

        if train_x.shape[0] % batch_size != 0:
            warnings.warn(
                "Number of samples not evenly divisible by batch size."
                " Smaller batch will occur."
            )
        if not self.initialized:
            self.initialize_weights(train_x[0:batch_size].shape)

        # Prep array on gpu
        batch_x_array = cp.zeros((batch_size, *train_x.shape[1:]))
        batch_y_array = cp.zeros((batch_size, *train_y.shape[1:]))

        for epoch in range(self.optimizer.epochs, self.optimizer.epochs + epochs):
            print(f"Epoch : {self.optimizer.epochs+1}")
            ids = list(range(0, train_x.shape[0]))
            if shuffle:
                list_shuffle(ids)
            for batch_idx in tqdm(range(0, train_x.shape[0], batch_size)):
                batch_x_array = cp.array(
                    train_x[ids[batch_idx : batch_idx + batch_size]]
                )
                batch_y_array = cp.array(
                    train_y[ids[batch_idx : batch_idx + batch_size]]
                )
                self.backward(
                    batch_x_array,
                    batch_y_array,
                )
            self.optimizer.epochs += 1
            print(
                f"Training Loss : {np.mean(np.mean(np.array(self.history['training_loss'][epoch])))}"
            )
            if not val_x is None and not val_y is None:
                if self.optimizer.epochs not in self.history["validation_loss"].keys():
                    self.history["validation_loss"][self.optimizer.epochs] = []

                for batch_idx in range(0, val_x.shape[0], batch_size):
                    batch_x_array = cp.array(val_x[batch_idx : batch_idx + batch_size])
                    batch_y_array = cp.array(val_y[batch_idx : batch_idx + batch_size])

                    prediction = self.forward(batch_x_array, training=False)
                    for a in (
                        self.optimizer.loss.forward(prediction, batch_y_array)
                        .get()
                        .tolist()
                    ):
                        self.history["validation_loss"][self.optimizer.epochs].append(a)
                print(
                    f"Validation Loss : {np.mean(np.mean(np.array(self.history['validation_loss'][self.optimizer.epochs])))}"
                )

    def clear_gradients(self):
        """Reset all gradients to 0"""
        for layer in self.layers:
            if layer.has_weights:
                layer.d_weights = cp.zeros(layer.d_weights.shape, dtype=cp.float32)
            if layer.has_bias:
                layer.d_bias = cp.zeros(layer.d_bias.shape, dtype=cp.float32)

    @abstractmethod
    def forward(
        self, inputs: cp.typing.NDArray[cp.float32]
    ) -> cp.typing.NDArray[cp.float32]:
        """Forward pass of model

        Args:
            inputs (NDArray [cp.float32]): inputs to compute
        Returns:
            NDArray [cp.float32]: model output
        """

    @abstractmethod
    def backward(
        self,
        inputs: cp.typing.NDArray[cp.float32],
        targets: cp.typing.NDArray[np.float32],
    ):
        """Backward pass of model

        Args:
            inputs (NDArray [cp.float32]): Inputs to pass through model
            targets (NDArray [cp.float32]): Targets for loss computation
            optimizer (Optimizer): Optimizer object for weight update computation
        """

    @abstractmethod
    def initialize_weights(self, input_shape: Tuple[int, ...]):
        """Initialize weights of model for given input shape

        Args:
            input_shape (Tuple[int,...]): Shape of inputs.
        """

    def predict(
        self, inputs: cp.typing.NDArray[cp.float32], batch_size
    ) -> np.typing.NDArray[np.float32]:
        outputs = np.zeros((inputs.shape[0], *self.layers[-1].output_shape[1:]))
        for batch_idx in range(0, inputs.shape[0], batch_size):
            batch_x_array = cp.array(inputs[batch_idx : batch_idx + batch_size])
            outputs[batch_idx : batch_idx + batch_size] = self.forward(
                batch_x_array, training=False
            ).get()
        return outputs
