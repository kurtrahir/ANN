"""Generic object for loss functions
"""


class Loss:
    """Generic Loss Object"""

    def __init__(self, forward, backward):
        self.forward = forward
        self.backward = backward
