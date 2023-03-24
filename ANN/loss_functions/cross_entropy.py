"""Categorical Cross Entropy Implementation"""


import numpy as np

from ANN.loss_functions.loss import Loss


class CrossEntropy(Loss):
    def __init__(self):
        def forward(pred, true):
            return np.sum(-true * np.log(pred + 1e-15), axis=1, keepdims=True)

        def backward(pred, true):
            return -true / (pred + 1e-15)

        Loss.__init__(self, forward=forward, backward=backward)
