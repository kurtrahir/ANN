"""Generic Model Interface"""
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from ANN import Optimizer, SGD
class Model:
    def __init__(
            self,
            forward : Callable[[NDArray[np.float32]], NDArray[np.float32]],
            backward : Callable[[NDArray[np.float32],NDArray[np.float32], Optimizer],],
            optimizer : Optimizer = SGD(0.1)
        ):
        self.forward = forward
        self.backward = backward
        self.optimizer = optimizer

        def train(
                self,
                inputs : NDArray[np.float32],
                targets : NDArray[np.float32],
                epochs : int,
                batch_size : int
            ):
            if inputs.shape[0] != targets.shape[0]:
                raise ValueError(
                    f"Mismatched number of samples in inputs and targets: \
                    {inputs.shape[0]} != {targets.shape[0]}"
                )

            if inputs.shape[0] % batch_size != 0:
                raise Warning(
                    "Number of samples not evenly divisible by batch size.\
                    Smaller batch will occur."
                )

            for _ in range(epochs):
                for batch_idx in range(batch_size, inputs.shape[0], batch_size):
                    self.backward(
                        inputs[batch_idx-batch_size:batch_idx],
                        targets[batch_idx-batch_size:batch_idx],
                        self.optimizer.get_update()
                    )
                optimizer.epochs += 1
