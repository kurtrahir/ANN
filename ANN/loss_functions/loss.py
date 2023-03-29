"""Generic object for loss functions
"""


from abc import ABC, abstractmethod

import cupy as np
from cupy.typing import NDArray


class Loss(ABC):
    """Generic Loss Object"""

    @abstractmethod
    def forward(self, pred: NDArray[np.float32], true: NDArray[np.float32]):
        """Calculate loss between pred and true

        Args:
            pred (NDArray[np.float32]): Predicted values
            true (NDArray[np.float32]): True values
        """

    @abstractmethod
    def backward(self, pred: NDArray[np.float32], true: NDArray[np.float32]):
        """Calculate partial loss derivative with regards to predicted values

        Args:
            pred (NDArray[np.float32]): Predicted values
            true (NDArray[np.float32]): True values
        """
