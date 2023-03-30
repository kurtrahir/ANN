"""Sigmoid activation function
"""
import cupy as cp
from cupy.typing import NDArray

from ANN.activation_functions import Activation


class Sigmoid(Activation):
    """Sigmoid activation function"""

    def __init__(self):
        Activation.__init__(self)

    def forward(self, pre_activation: NDArray[cp.float32]) -> NDArray[cp.float32]:
        self.activations = 1 / (1 + cp.exp(-pre_activation))
        return self.activations

    def backward(
        self, partial_loss_derivative: NDArray[cp.float32]
    ) -> NDArray[cp.float32]:
        local_gradient = self.activations * (1 - self.activations)
        return local_gradient * partial_loss_derivative
