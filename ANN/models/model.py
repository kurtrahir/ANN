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
        self.history = {"Losses" : {}}

    def train(
            self,
            inputs : NDArray[np.float32],
            targets : NDArray[np.float32],
            epochs : int,
            batch_size : int
        ):
        """Method to train network on provided dataset."""
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Mismatched number of samples in inputs and targets: \
                {inputs.shape[0]} != {targets.shape[0]}"
            )

        if inputs.shape[0] % batch_size != 0:
            warnings.warn(
                "Number of samples not evenly divisible by batch size.\
                Smaller batch will occur."
            )

        loss_function = self.layers[-1].loss_function
        for _ in range(epochs):
            print(f"Epoch : {self.optimizer.epochs+1}")
            for batch_idx in tqdm(range(batch_size, inputs.shape[0], batch_size)):
                self.backward(
                    inputs[batch_idx-batch_size:batch_idx],
                    targets[batch_idx-batch_size:batch_idx]
                )
            self.optimizer.epochs += 1
            loss = loss_function.forward(self.forward(inputs), targets)
            self.history["Losses"][self.optimizer.epochs] = loss
            print(f"Loss : {np.mean(np.mean(loss, axis = 1))}")
