"""Categorical Cross Entropy Implementation"""


import cupy as np

from ANN.loss_functions.loss import Loss


class CrossEntropy(Loss):
    def __init__(self):
        Loss.__init__(self)

    def forward(self, pred, true):
        return np.sum(
            -true * np.log(pred / np.sum(pred, axis=-1, keepdims=True) + 1e-15), axis=1
        )

    def backward(self, pred, true):
        inverted = np.abs(true - 1)
        prob_one = np.sum(pred * true, axis=-1, keepdims=True)
        p_sum = np.sum(pred, axis=-1, keepdims=True)
        return inverted / p_sum - true * np.sum(
            inverted * pred, axis=-1, keepdims=True
        ) / (prob_one * p_sum)
