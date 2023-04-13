"""Stochastic Gradient Descent optimizer"""
import cupy as cp
from cupy.typing import NDArray

from ANN.loss_functions.loss import Loss
from ANN.optimizers import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent Optimizer"""

    def __init__(self, loss: Loss, learning_rate: cp.float32):
        self.learning_rate = learning_rate
        Optimizer.__init__(self, loss=loss)

    def backward(
        self, model, inputs: NDArray[cp.float32], targets: NDArray[cp.float32]
    ):
        outputs = model.forward(inputs, training=True)

        d_loss = self.loss.backward(outputs, targets)

        for layer in model.layers[::-1]:
            d_loss = layer.backward(d_loss)

        # Update weights, averaging over batch and multiplying with learning rate.
        for layer in model.layers:
            if layer.has_weights:
                layer.weights -= cp.divide(
                    cp.multiply(self.learning_rate, layer.d_weights),
                    inputs.shape[0],
                )
            if layer.has_bias:
                layer.bias -= cp.divide(
                    cp.multiply(self.learning_rate, layer.d_bias), inputs.shape[-1]
                )

        return self.loss.forward(outputs, targets)
