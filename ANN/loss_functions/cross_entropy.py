"""Categorical Cross Entropy Implementation"""


import cupy as cp

from ANN.loss_functions.loss import Loss


class CrossEntropy(Loss):
    def __init__(self):
        Loss.__init__(self)

    def forward(self, pred, true):
        return cp.sum(
            -true * cp.log(pred / cp.sum(pred, axis=-1, keepdims=True) + 1e-15), axis=1
        )

    def backward(self, pred, true):
        inverted = cp.abs(true - 1)
        prob_one = cp.sum(pred * true, axis=-1, keepdims=True)
        p_sum = cp.sum(pred, axis=-1, keepdims=True)
        return inverted / p_sum - true * cp.sum(
            inverted * pred, axis=-1, keepdims=True
        ) / (prob_one * p_sum)
