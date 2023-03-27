"""TanH activation function"""

import cupy as np

from ANN.activation_functions import Activation


class TanH(Activation):
    def __init__(self):
        Activation.__init__(self)

    def forward(self, pre_activation):
        self.activations = np.tanh(pre_activation)
        return self.activations

    def backward(self, partial_loss_derivative):
        local_gradient = 1 - self.activations**2
        return local_gradient * partial_loss_derivative
