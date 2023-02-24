"""Stochastic Gradient Descent optimizer"""
import numpy as np
from numpy.typing import NDArray
from ANN.optimizers import Optimizer

class SGD(Optimizer):
    """Stochastic Gradient Descent Optimizer
    """
    def __init__(self, learning_rate):

        self.learning_rate = learning_rate

        def backward(
            model,
            inputs : NDArray[np.float32],
            targets : NDArray[np.float32]
        ):

            n_samples = inputs.shape[0]
            gradients = [
                    np.zeros(layer.weights.shape) for layer in model.layers
                ]

            for sample_idx in range(n_samples):
                _ = model.forward(inputs[sample_idx].reshape(1,-1))
                temp_t = targets[sample_idx]
                for layer_idx in range(1,len(model.layers)):
                    temp_t = model.layers[-layer_idx].backward(temp_t)
                    gradients[-layer_idx] += model.layers[-layer_idx].d_weights

            for i,layer in enumerate(model.layers):
                layer.weights -= self.learning_rate * gradients[i] / n_samples

        Optimizer.__init__(
            self,
            backward=backward
        )
