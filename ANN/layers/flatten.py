"""Flatten layer implementation"""


import operator
from functools import reduce

from ANN.layers.layer import Layer


class Flatten(Layer):
    """Flatten layer implementation"""

    def __init__(self):
        self.input_shape = None

        def forward(inputs):
            self.input_shape = inputs.shape
            return inputs.reshape(inputs.shape[0], -1)

        def backward(inputs):
            return inputs.reshape(self.input_shape)

        def initialize_layer(input_shape):
            self.input_shape = input_shape
            self.output_shape = (1, reduce(operator.mul, input_shape, 1))

        Layer.__init__(
            self,
            forward=forward,
            backward=backward,
            initialize_weights=initialize_layer,
            has_weights=False,
            weights=None,
            input_shape=None,
            output_shape=None,
        )
