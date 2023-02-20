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
        Activation.__init__(
            self,
            lambda x: np.where(x > 0, x, 0),
            lambda x: (x > 0).astype(int)
        )
