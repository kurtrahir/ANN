"""Generic Model Interface"""
import warnings
from abc import ABC, abstractmethod
from random import shuffle as list_shuffle
from typing import Tuple

import cupy as np
from cupy.typing import NDArray
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
        train_x: NDArray[np.float32],
        train_y: NDArray[np.float32],
        epochs: int,
        batch_size: int,
        val_x: NDArray[np.float32] = None,
        val_y: NDArray[np.float32] = None,
        shuffle: bool = True,
    ):
        """Method to train network on provided dataset."""
        if train_x.shape[0] != train_y.shape[0]:
            raise ShapeError(
                f"Mismatched number of samples in training inputs and targets: \
                {train_x.shape[0]} != {train_y.shape[0]}"
            )
        set_list = [(train_x, "training")]
        if not val_x is None and not val_y is None:
            if val_x.shape[0] != val_y.shape[0]:
                raise ShapeError(
                    f"Mismatched number of samples in validation inputs and targets: \
                    {val_x.shape[0]} != {val_y.shape[0]}"
                )
            set_list.append((val_x, "validation"))

        for input_x, set_name in set_list:
            if len(input_x.shape) < 2:
                raise ShapeError(
                    f"Model expected {set_name} input vector of shape (n_samples, (input_shape))) \
                    received {input_x.shape} instead."
                )

        if train_x.shape[0] % batch_size != 0:
            warnings.warn(
                "Number of samples not evenly divisible by batch size.\
                Smaller batch will occur."
            )
        if not self.initialized:
            self.initialize_weights(train_x[0:batch_size].shape)

        for epoch in range(self.optimizer.epochs, self.optimizer.epochs + epochs):
            print(f"Epoch : {self.optimizer.epochs+1}")
            ids = list(range(0, train_x.shape[0]))
            if shuffle:
                list_shuffle(ids)
            for batch_idx in tqdm(range(0, train_x.shape[0], batch_size)):
                self.backward(
                    train_x[ids[batch_idx : batch_idx + batch_size]],
                    train_y[ids[batch_idx : batch_idx + batch_size]],
                )
            self.optimizer.epochs += 1
            print(
                f"Training Loss : {np.mean(np.mean(np.array(self.history['training_loss'][epoch]), axis = 1)).get()}"
            )
            if not val_x is None and not val_y is None:
                validation_loss = self.optimizer.loss.forward(
                    self.forward(val_x), val_y
                )
                validation_loss = np.array(validation_loss).reshape(val_x.shape[0], -1)
                self.history["validation_loss"][self.optimizer.epochs] = validation_loss
                print(
                    f"Validation Loss : {np.mean(np.mean(validation_loss, axis = 1))}"
                )

    def clear_gradients(self):
        """Reset all gradients to 0"""
        for layer in self.layers:
            if layer.has_weights:
                layer.d_weights = np.zeros(layer.d_weights.shape, dtype=np.float32)
            if layer.has_bias:
                layer.d_bias = np.zeros(layer.d_bias.shape, dtype=np.float32)

    @abstractmethod
    def forward(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Forward pass of model

        Args:
            inputs (NDArray[np.float32]): inputs to compute
        Returns:
            NDArray[np.float32]: model output
        """

    @abstractmethod
    def backward(self, inputs: NDArray[np.float32], targets: NDArray[np.float32]):
        """Backward pass of model

        Args:
            inputs (NDArray[np.float32]): Inputs to pass through model
            targets (NDArray[np.float32]): Targets for loss computation
            optimizer (Optimizer): Optimizer object for weight update computation
        """

    @abstractmethod
    def initialize_weights(self, input_shape: Tuple[int, ...]):
        """Initialize weights of model for given input shape

        Args:
            input_shape (Tuple[int,...]): Shape of inputs.
        """
