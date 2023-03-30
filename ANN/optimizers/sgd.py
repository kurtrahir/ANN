"""Stochastic Gradient Descent optimizer"""
import cupy as np
from cupy.typing import NDArray

from ANN.loss_functions.loss import Loss
from ANN.optimizers import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent Optimizer"""

    def __init__(self, loss: Loss, learning_rate: np.float32):
        self.learning_rate = learning_rate

        def backward(model, inputs: NDArray[np.float32], targets: NDArray[np.float32]):
            outputs = model.forward(inputs)
            d_loss = self.loss.backward(outputs, targets)

            if not self.epochs in model.history["training_loss"].keys():
                model.history["training_loss"][self.epochs] = []

            model.history["training_loss"][self.epochs].append(
                self.loss.forward(outputs, targets)
            )

            for layer in model.layers[::-1]:
                d_loss = layer.backward(d_loss)

            # Update weights, averaging over batch and multiplying with learning rate.
            for layer in model.layers:
                if layer.has_weights:
                    layer.weights -= np.divide(
                        np.multiply(self.learning_rate, layer.d_weights),
                        inputs.shape[0],
                    )
                if layer.has_bias:
                    layer.bias -= np.divide(
                        np.multiply(self.learning_rate, layer.d_bias), inputs.shape[-1]
                    )

        Optimizer.__init__(self, loss=loss, backward=backward)
