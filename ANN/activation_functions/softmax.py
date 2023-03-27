"""Softmax implementation
"""


import cupy as np

from ANN.activation_functions.activation import Activation


class Softmax(Activation):
    """Softmax Implementation

    Args:
        Activation (Activation): General Activation Function Class
    """

    def __init__(self):
        Activation.__init__(self)

    def forward(self, pre_activation):
        exp_x = np.exp(pre_activation - np.max(pre_activation, axis=1, keepdims=True))
        self.activations = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.activations

    def backward(self, partial_loss_derivative):
        jacobian_matrices = np.empty(
            (
                partial_loss_derivative.shape[0],
                self.activations.shape[1],
                self.activations.shape[1],
            )
        )
        for i in range(partial_loss_derivative.shape[0]):
            jacobian_matrices[i] = np.diag(self.activations[i]) - np.outer(
                self.activations[i], self.activations[i]
            )
        return np.einsum("ijk,ik->ij", jacobian_matrices, partial_loss_derivative)
