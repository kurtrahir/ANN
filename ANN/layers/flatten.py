"""Flatten layer implementation"""

from ANN.layers.layer import Layer


class Flatten(Layer):
    """Flatten layer implementation"""

    def __init__(self):
        self.input_shape = "Unknown"

        def forward(inputs):
            self.input_shape = inputs.shape
            return inputs.reshape(1, -1)

        def backward(inputs):
            return inputs.reshape(self.input_shape)

        Layer.__init__(
            self, forward=forward, backward=backward, d_weights=None, has_weights=False
        )
