"""Binary Cross Entropy Loss Function"""

import numpy as np
from ANN.loss_functions import Loss

class BinaryCrossEntropy(Loss):
    """Binary Cross Entropy Loss Function"""
    def __init__(self):
        Loss.__init__(
            self,
            lambda pred, true : np.multiply(-true, np.log2(pred+1e-15)) - np.multiply(1-true, np.log2(1-pred+1e-15)),
            lambda pred, true : np.divide(-true,pred+1e-15) + np.divide(1-true, 1-pred+1e-15)
        )
