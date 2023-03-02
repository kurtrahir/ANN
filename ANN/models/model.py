"""Generic Model Interface"""
import warnings
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ANN.errors import ShapeError
from ANN.layers import Layer
from ANN.optimizers.optimizer import Optimizer


class Model:
    """Generic Model Interface"""

    def __init__(
        self,
        forward: Callable[[NDArray[np.float32]], NDArray[np.float32]],
        backward: Callable[[NDArray[np.float32], NDArray[np.float32], Optimizer], None],
        layers: list[Layer],
        optimizer: Optimizer,
    ):
        self.forward = forward
        self.backward = backward
        self.optimizer = optimizer
        self.layers = layers
        self.history = {"training_loss": {}, "validation_loss": {}}

    def train(
        self,
        train_x: NDArray[np.float32],
        train_y: NDArray[np.float32],
        epochs: int,
        batch_size: int,
        val_x: NDArray[np.float32] = None,
        val_y: NDArray[np.float32] = None,
    ):
        """Method to train network on provided dataset."""
        if train_x.shape[0] != train_y.shape[0]:
            raise ShapeError(
                f"Mismatched number of samples in training inputs and targets: \
                {train_x.shape[0]} != {train_y.shape[0]}"
            )
        if not val_x is None and not val_y is None:
            if val_x.shape[0] != val_y.shape[0]:
                raise ShapeError(
                    f"Mismatched number of samples in validation inputs and targets: \
                    {val_x.shape[0]} != {val_y.shape[0]}"
                )

        for input_x, set_name in ((train_x, "training"), (val_x, "validation")):
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

        for _ in range(epochs):
            print(f"Epoch : {self.optimizer.epochs+1}")
            for batch_idx in tqdm(range(0, train_x.shape[0], batch_size)):
                self.backward(
                    train_x[batch_idx : batch_idx + batch_size],
                    train_y[batch_idx : batch_idx + batch_size],
                )
            self.optimizer.epochs += 1
            training_loss = self.optimizer.loss.forward(self.forward(train_x), train_y)
            self.history["training_loss"][self.optimizer.epochs] = training_loss
            print(f"Training Loss : {np.mean(np.mean(training_loss, axis = 1))}")
            if not val_x is None and not val_y is None:
                validation_loss = self.optimizer.loss.forward(
                    self.forward(val_x), val_y
                )
                self.history["validation_loss"][self.optimizer.epochs] = validation_loss
                print(
                    f"Validation Loss : {np.mean(np.mean(validation_loss, axis = 1))}"
                )
