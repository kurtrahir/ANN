"""Object definition for neural net activation functions.
"""

from abc import abstractmethod

import cupy as cp
from numpy import ndarray


class Activation:
    """Object definition for neural net activation functions."""

    def __init__(self):
        """Initialize activation function object"""
        # Variable to store activations on forward pass for access on backwards pass
        self.activations = None

    @abstractmethod
    def forward(self, pre_activation: ndarray[cp.float32]) -> ndarray[cp.float32]:
        """Calculate forward pass of activation function

        Args:
            pre_activation (ndarray [cp.float32]): Preactivation values
        Returns:
            ndarray [cp.float32]: Activation values
        """

    @abstractmethod
    def backward(
        self, partial_loss_derivative: ndarray[cp.float32]
    ) -> ndarray[cp.float32]:
        """Calculate backward pass using given partial loss derivative
           (combines partial loss derivative with local gradient)

        Args:
            partial_loss_derivative (ndarray [cp.float32]): Partial loss derivative values
        Returns:
            ndarray [cp.float32]: Partial derivative (da/dz * dL/da)
        """
