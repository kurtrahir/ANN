"""Adam optimizer"""

import cupy as cp
from cupy.typing import NDArray

from ANN.loss_functions import Loss
from ANN.optimizers.optimizer import Optimizer


class Adam(Optimizer):
    """Implementation of the Adam optimizer"""

    def __init__(
        self, loss: Loss, beta_1=0.9, beta_2=0.999, learning_rate=1e-3, epsilon=1e-15
    ):
        self.momentums = {}
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        def backward(model, inputs: NDArray[cp.float32], targets: NDArray[cp.float32]):
            if not self.momentums:
                for i, layer in enumerate(model.layers):
                    if layer.has_weights:
                        self.momentums[i] = {
                            "first_order": cp.zeros((layer.weights.shape)),
                            "second_order": cp.zeros((layer.weights.shape)),
                        }
                    if layer.has_bias:
                        self.momentums[f"bias_{i}"] = {
                            "first_order": cp.zeros((layer.bias.shape)),
                            "second_order": cp.zeros((layer.bias.shape)),
                        }

            n_samples = inputs.shape[0]
            # Accumulate gradients
            outputs = model.forward(inputs, training=True)
            d_loss = self.loss.backward(outputs, targets)

            if not self.epochs in model.history["training_loss"].keys():
                model.history["training_loss"][self.epochs] = []

            for a in self.loss.forward(outputs, targets).get().tolist():
                model.history["training_loss"][self.epochs].append(a)
            for layer in model.layers[::-1]:
                d_loss = layer.backward(d_loss)
            # Update weights by obtaining adam update term, averaging over batch
            # and multiplying by learning rate.
            for i, layer in enumerate(model.layers):
                if layer.has_weights:
                    layer.weights -= self.learning_rate * self.get_update(
                        layer.d_weights / n_samples, i
                    )
                if layer.has_bias:
                    layer.bias -= self.learning_rate * self.get_update(
                        layer.d_bias / n_samples, f"bias_{i}"
                    )

        Optimizer.__init__(self, loss=loss, backward=backward)

    def update_momentums(self, gradients, layer_idx):
        """Update first order and second order momentums"""
        self.momentums[layer_idx]["first_order"] *= self.beta_1
        self.momentums[layer_idx]["first_order"] += (1 - self.beta_2) * gradients
        self.momentums[layer_idx]["second_order"] *= self.beta_2
        self.momentums[layer_idx]["second_order"] += (1 - self.beta_2) * (
            gradients**2
        )

    def get_update(self, gradients, layer_idx):
        """Compute update term"""
        self.update_momentums(gradients, layer_idx)
        first_order = self.momentums[layer_idx]["first_order"] / (1 - self.beta_1)
        second_order = self.momentums[layer_idx]["second_order"] / (1 - self.beta_2)
        return (first_order) / (cp.sqrt(second_order) + self.epsilon)
