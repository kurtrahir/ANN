"""Softmax implementation
"""


import cupy as cp

from ANN.activation_functions.activation import Activation


class Softmax(Activation):
    """Softmax Implementation

    Args:
        Activation (Activation): General Activation Function Class
    """

    def __init__(self):
        Activation.__init__(self)

    def forward(self, pre_activation):
        exp_x = cp.exp(pre_activation - cp.max(pre_activation, axis=1, keepdims=True))
        self.activations = exp_x / cp.sum(exp_x, axis=1, keepdims=True)
        return self.activations

    def backward(self, partial_loss_derivative):
        jacobian_matrices = cp.empty(
            (
                partial_loss_derivative.shape[0],
                self.activations.shape[1],
                self.activations.shape[1],
            )
        )
        for i in range(partial_loss_derivative.shape[0]):
            jacobian_matrices[i] = cp.diag(self.activations[i]) - cp.outer(
                self.activations[i], self.activations[i]
            )
        return cp.einsum("ijk,ik->ij", jacobian_matrices, partial_loss_derivative)
