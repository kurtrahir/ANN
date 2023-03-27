"""Sigmoid activation function
"""
import cupy as np
from numpy.typing import NDArray

from ANN.activation_functions import Activation


class Sigmoid(Activation):
    """Sigmoid activation function"""

    def __init__(self):
        Activation.__init__(self)

    def forward(self, pre_activation: NDArray[np.float32]) -> NDArray[np.float32]:
        self.activations = 1 / (1 + np.exp(-pre_activation))
        return self.activations

    def backward(
        self, partial_loss_derivative: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        local_gradient = self.activations * (1 - self.activations)
        return local_gradient * partial_loss_derivative
