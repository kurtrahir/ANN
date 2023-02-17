"""ReLu implementation
"""
from Activation import Activation


class ReLu(Activation):
    """Implementation of relu activation function

    Args:
        Activation (np.float32): _description_
    """

    def __init__(self):
        Activation.__init__(
            self,
            lambda x: x if x > 0 else 0,
            lambda x: 1 if x > 0 else 0,
        )
