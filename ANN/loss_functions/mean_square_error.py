"""MSE loss object implementation
"""
import cupy as np

from ANN.loss_functions import Loss


class MSE(Loss):
    """MSE loss object

    Args:
        Loss (Loss): Implement Loss object
    """

    def __init__(self):
        Loss.__init__(self, lambda x, y: np.square(np.subtract(x, y)) / 2, np.subtract)
