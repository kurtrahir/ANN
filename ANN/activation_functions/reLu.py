"""ReLu implementation
"""
import numpy as np

from ANN.activation_functions import Activation


class ReLu(Activation):
    """Implementation of relu activation function

    Args:
        Activation (np.float32): _description_
    """

    def __init__(self):
        Activation.__init__(self)

    def forward(self, pre_activation):
        self.activations = np.where(pre_activation > 0, pre_activation, 0)
        return self.activations

    def backward(self, partial_loss_derivative):
        local_gradient = (self.activations > 0).astype(int)
        return local_gradient * partial_loss_derivative
