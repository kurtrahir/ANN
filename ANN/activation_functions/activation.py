"""Object definition for neural net activation functions.
"""


class Activation:
    """Object definition for neural net activation functions."""

    def __init__(self, forward, backward):
        """Initialize activation function

        Args:
            forward (callable[np.float32] : np.float32): Forward pass of activation function
            backward (callable[np.flaot32] : np.float32):
                Backward pass (derivative) of activation function.
        """
        self.forward = forward
        self.backward = backward
