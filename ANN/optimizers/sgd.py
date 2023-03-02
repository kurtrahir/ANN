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
            gradients = []
            for layer in model.layers:
                if layer.has_weights:
                    gradients.append(np.zeros(layer.d_weights.shape))

            # Iterate over batch
            for sample_idx in range(n_samples):
                # Compute forward pass obtaining prediction, and
                # setting internal variables required for backward pass
                pred = model.forward(
                    inputs[sample_idx].reshape(1, *inputs[sample_idx].shape)
                )
                # Obtain loss derivative (error term for output layer)
                temp_t = self.loss.backward(pred, targets[sample_idx])
                # Backpropagate, accumulating gradients
                gradient_idx = -1
                for layer_idx in range(1, len(model.layers) + 1):
                    temp_t = model.layers[-layer_idx].backward(temp_t)
                    if model.layers[-layer_idx].has_weights:
                        gradients[gradient_idx] += model.layers[-layer_idx].d_weights
                        gradient_idx -= 1

            # Update weights, averaging over batch and multiplying with learning rate.
            gradient_idx = 0
            for i, _ in enumerate(model.layers):
                if model.layers[i].has_weights:
                    model.layers[i].weights -= (
                        self.learning_rate * gradients[gradient_idx] / n_samples
                    )
                    gradient_idx += 1

        Optimizer.__init__(self, loss=loss, backward=backward)
