"""Generic Optimizer interface
"""
from typing import Callable
import numpy as np
from numpy.typing import NDArray

from ANN import Model

class Optimizer():
    def __init__(
        self,
        get_update : Callable[[Model], NDArray[np.float32]]
    ):
        self.get_update = get_update
        self.epochs = 0
