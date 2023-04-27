"""Categorical Cross Entropy Implementation"""


import cupy as cp

from ANN.loss_functions.loss import Loss


class CrossEntropy(Loss):
    def __init__(self):
        Loss.__init__(self)

    def backward(self, pred, true):
        inverted = cp.abs(true - 1)
        p_true = cp.sum(pred * true, axis=-1, keepdims=True)
        p_sum = cp.sum(pred, axis=-1, keepdims=True)
        p_zeroes = cp.sum(inverted * pred, axis=-1, keepdims=True)
        return inverted / p_sum - true * p_zeroes / (p_true * p_sum)
