"""MSE loss object implementation
"""
import numpy as np
from ANN.loss_functions import Loss


class SquareError(Loss):
    """MSE loss object

    Args:
        Loss (Loss): Implement Loss object
    """
    def __init__(self):
        Loss.__init__(
            self,
            lambda x,y: np.square(np.subtract(x,y)),
            lambda x,y: np.multiply(2,(np.subtract(x,y)))
        )
