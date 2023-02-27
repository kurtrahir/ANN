"""Neural Network layer object"""


class Layer:
    """Neural Network Layer Object"""

    def __init__(self, forward, backward, d_weights):
        self.forward = forward
        self.backward = backward
        self.d_weights = d_weights
