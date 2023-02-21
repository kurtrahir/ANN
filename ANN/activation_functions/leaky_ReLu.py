"""Leaky ReLu implementation"""

import numpy as np
from ANN import Activation

class LeakyReLu(Activation):
    """Leaky ReLu class"""
    def __init__(self, a : int = 0.01):
        Activation.__init__(
            self,
            forward = lambda x : np.where(x > 0, x, a * x),
            backward = lambda x : np.where(x > 0, 1, a)
        )
