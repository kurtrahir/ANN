"""Categorical Cross Entropy Implementation"""


import cupy as np

from ANN.activation_functions.softmax import Softmax
from ANN.loss_functions.loss import Loss


class CategoricalCrossEntropy(Loss):
    def __init__(self):
        self.softmax = Softmax()

        def forward(pred, true):
            return np.sum(
                -true * np.log(self.softmax.forward(pred) + 1e-15),
                axis=1,
                keepdims=True,
            )

        def backward(pred, true):
            return pred - true

        Loss.__init__(self, forward=forward, backward=backward)
