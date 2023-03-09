"""Stochastic Gradient Descent optimizer"""
import numpy as np
from numpy.typing import NDArray

from ANN.loss_functions.loss import Loss
from ANN.optimizers import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent Optimizer"""

    def __init__(self, loss: Loss, learning_rate: np.float32):
        self.learning_rate = learning_rate

        def backward(model, inputs: NDArray[np.float32], targets: NDArray[np.float32]):
            n_samples = inputs.shape[0]

            outputs = model.forward(inputs)
            d_loss = self.loss.backward(outputs, targets)
            for layer in model.layers[::-1]:
                d_loss = layer.backward(d_loss)

            # Update weights, averaging over batch and multiplying with learning rate.
            for layer in model.layers:
                if layer.has_weights:
                    layer.weights -= self.learning_rate * layer.d_weights / n_samples
                if layer.has_bias:
                    layer.bias -= self.learning_rate * layer.d_bias / n_samples

            model.clear_gradients()

        Optimizer.__init__(self, loss=loss, backward=backward)
