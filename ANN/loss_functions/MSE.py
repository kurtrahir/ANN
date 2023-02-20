"""MSE loss object implementation
"""
import numpy as np
from ANN import Loss


class MSE(Loss):
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
