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
            gradients = [np.zeros(layer.weights.shape) for layer in model.layers]

            # Iterate over batch
            for sample_idx in range(n_samples):
                # Compute forward pass obtaining prediction, and
                # setting internal variables required for backward pass
                pred = model.forward(inputs[sample_idx].reshape(1, -1))
                # Obtain loss derivative (error term for output layer)
                temp_t = self.loss.backward(pred, targets[sample_idx])
                # Backpropagate, accumulating gradients
                for layer_idx in range(1, len(model.layers) + 1):
                    temp_t = model.layers[-layer_idx].backward(temp_t)
                    gradients[-layer_idx] += model.layers[-layer_idx].d_weights

            # Update weights, averaging over batch and multiplying with learning rate.
            for i, _ in enumerate(model.layers):
                model.layers[i].weights -= self.learning_rate * gradients[i] / n_samples

        Optimizer.__init__(self, loss=loss, backward=backward)
