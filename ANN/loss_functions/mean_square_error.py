"""MSE loss object implementation
"""
import cupy as cp

from ANN.loss_functions import Loss


class MSE(Loss):
    """MSE loss object

    Args:
        Loss (Loss): Implement Loss object
    """

    def __init__(self):
        Loss.__init__(self)

    def forward(self, pred, true):
        return cp.mean(cp.square(cp.subtract(pred, true)), axis=-1)

    def backward(self, pred, true):
        return 2 / pred.shape[-1] * (pred - true)
