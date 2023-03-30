"""Leaky ReLu implementation"""

import cupy as cp

from ANN.activation_functions import Activation


class LeakyReLu(Activation):
    """Leaky ReLu class"""

    def __init__(self, epsilon: int = 0.01):
        self.epsilon = epsilon
        Activation.__init__(self)

    def forward(self, pre_activation):
        self.activations = cp.where(
            pre_activation > 0, pre_activation, pre_activation * self.epsilon
        )
        return self.activations

    def backward(self, partial_loss_derivative):
        local_gradient = cp.where(self.activations > 0, 1, self.epsilon)
        return local_gradient * partial_loss_derivative
