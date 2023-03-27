"""Linear unit implementation
"""
import cupy as np

from ANN.activation_functions import Activation


class Linear(Activation):
    """Implementation of linear activation function

    Args:
        Activation (np.float32): Generic activation function type
    """

    def __init__(self):
        self.output = None
        Activation.__init__(self)

    def forward(self, pre_activation):
        self.activations = pre_activation
        return pre_activation

    def backward(self, partial_loss_derivative):
        local_gradient = np.invert(np.isclose(self.activations, 0))
        return local_gradient * partial_loss_derivative
