"""ReLu implementation
"""
import cupy as cp

from ANN.activation_functions import Activation


class ReLu(Activation):
    """Implementation of relu activation function

    Args:
        Activation (cp.float32): _description_
    """

    def __init__(self):
        Activation.__init__(self)

    def forward(self, pre_activation):
        self.activations = cp.where(pre_activation > 0, pre_activation, 0)
        return self.activations

    def backward(self, partial_loss_derivative):
        local_gradient = (self.activations > 0).astype(int)
        return local_gradient * partial_loss_derivative
