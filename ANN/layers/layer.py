"""Neural Network layer object"""


class Layer:
    """Neural Network Layer Object"""

    def __init__(
        self,
        forward,
        backward,
        initialize_weights,
        has_weights,
        weights,
        input_shape,
        output_shape,
        has_bias=False,
    ):
        self.forward = forward
        self.backward = backward
        self.initialize_weights = initialize_weights
        self.has_weights = has_weights
        self.has_bias = has_bias

        self.weights = weights
        self.input_shape = input_shape
        self.output_shape = output_shape
