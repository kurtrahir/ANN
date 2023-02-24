"""TanH activation function"""

import numpy as np
from ANN.activation_functions import Activation

class TanH(Activation):

    def __init__(self):
        Activation.__init__(
            self,
            forward = np.tanh,
            backward = lambda x : 1 - self.forward(x)**2
        )
