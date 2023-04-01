"""Flatten layer implementation"""


import operator
from functools import reduce

from ANN.layers.layer import Layer


class Flatten(Layer):
    """Flatten layer implementation"""

    def __init__(self):
        self.input_shape = None

        Layer.__init__(
            self,
            has_weights=False,
            has_bias=False,
            input_shape=None,
            output_shape=None,
        )

    def forward(self, inputs, training: bool = False):
        self.input_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, gradient):
        return gradient.reshape(self.input_shape)

    def initialize(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (
            input_shape[0],
            reduce(operator.mul, input_shape[1:], 1),
        )
