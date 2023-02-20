"""Linear unit implementation
"""
import numpy as np
from ANN.activation_functions import Activation


class Linear(Activation):
    """Implementation of relu activation function

    Args:
        Activation (np.float32): Generic activation function type
    """

    def __init__(self):
        Activation.__init__(
            self,
            lambda x: x,
            lambda x: np.invert(np.isclose(x, 0))
        )
