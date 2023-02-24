"""Sigmoid activation function
"""
import numpy as np
from ANN.activation_functions import Activation

class Sigmoid(Activation):
    """Sigmoid activation function"""
    def __init__(self):
        Activation.__init__(
            self,
            forward = lambda x : 1 / (1 + np.exp(-x)),
            backward = lambda x : self.forward(x) * (1 - self.forward(x))
        )
