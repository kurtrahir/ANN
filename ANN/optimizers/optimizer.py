"""Generic Optimizer interface
"""
from abc import abstractmethod
from typing import Callable

import cupy as cp
from cupy.typing import NDArray

from ANN.loss_functions.loss import Loss


class Optimizer:
    "Generic optimizer interface"

    def __init__(
        self,
        loss: Loss,
    ):
        self.loss = loss
        self.epochs = 0

    @abstractmethod
    def backward(inputs: NDArray[cp.float32], targets: NDArray[cp.float32]):
        """Update the weights of the model using the provided batch."""
