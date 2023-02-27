"""Generic Model Interface"""
import warnings
from typing import Callable
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
from ANN.optimizers import Optimizer
from ANN.layers import Layer


class Model:
    """Generic Model Interface"""
    def __init__(
            self,
            forward : Callable[[NDArray[np.float32]], NDArray[np.float32]],
            backward : Callable[[NDArray[np.float32],NDArray[np.float32], Optimizer], None],
            layers : list[Layer],
            optimizer : Optimizer
        ):
        self.forward = forward
        self.backward = backward
        self.optimizer = optimizer
        self.layers = layers
        self.history = {"training_loss": {}, "validation_loss": {}}

    def train(
            self,
            x : NDArray[np.float32],
            y : NDArray[np.float32],
            epochs : int,
            batch_size : int,
            val_x : NDArray[np.float32] = None,
            val_y : NDArray[np.float32] = None,
        ):
        """Method to train network on provided dataset."""
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Mismatched number of samples in inputs and targets: \
                {x.shape[0]} != {y.shape[0]}"
            )

        if x.shape[0] % batch_size != 0:
            warnings.warn(
                "Number of samples not evenly divisible by batch size.\
                Smaller batch will occur."
            )
            

        for _ in range(epochs):
            print(f"Epoch : {self.optimizer.epochs+1}")
            for batch_idx in tqdm(range(batch_size, x.shape[0], batch_size)):
                self.backward(
                    x[batch_idx-batch_size:batch_idx],
                    y[batch_idx-batch_size:batch_idx]
                )
            self.optimizer.epochs += 1
            training_loss = self.optimizer.loss.forward(self.forward(x), y)
            self.history["training_loss"][self.optimizer.epochs] = training_loss
            print(f"Training Loss : {np.mean(np.mean(training_loss, axis = 1))}")
            if not val_x is None and not val_y is None:
                validation_loss = self.optimizer.loss.forward(self.forward(val_x), val_y)
                self.history["validation_loss"][self.optimizer.epochs] = validation_loss
                print(
                    f"Validation Loss : {np.mean(np.mean(validation_loss, axis = 1))}")
