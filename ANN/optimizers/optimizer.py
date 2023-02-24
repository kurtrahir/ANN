"""Generic Optimizer interface
"""
from typing import Callable
import numpy as np
from numpy.typing import NDArray

class Optimizer():
    "Generic optimizer interface"
    def __init__(
        self,
        backward : Callable[[], NDArray[np.float32]],
    ):
        self.backward = backward
        self.epochs = 0
