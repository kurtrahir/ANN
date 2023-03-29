"""Generic Optimizer interface
"""
from typing import Callable

import cupy as np
from cupy.typing import NDArray

from ANN.loss_functions.loss import Loss


class Optimizer:
    "Generic optimizer interface"

    def __init__(
        self,
        loss: Loss,
        backward: Callable[[], NDArray[np.float32]],
    ):
        self.backward = backward
        self.loss = loss
        self.epochs = 0
