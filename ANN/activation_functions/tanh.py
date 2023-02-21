"""TanH activation function"""

import numpy as np
from ANN import Activation

class TanH(Activation):

    def __init__(self):
        Activation.__init__(
            self,
            forward = lambda x : np.tanh(x=x),
            backward = lambda x : 1 - self.forward(x)**2
        )
