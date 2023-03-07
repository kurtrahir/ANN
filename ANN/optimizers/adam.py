"""Adam optimizer"""

import numpy as np
from numpy.typing import NDArray

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

        def backward(model, inputs: NDArray[np.float32], targets: NDArray[np.float32]):
            if not self.momentums:
                for i, layer in enumerate(model.layers):
                    if layer.has_weights:
                        self.momentums[i] = {
                            "first_order": np.zeros((layer.weights.shape)),
                            "second_order": np.zeros((layer.weights.shape)),
                        }
                    if layer.has_bias:
                        self.momentums[f"bias_{i}"] = {
                            "first_order": np.zeros((layer.bias.shape)),
                            "second_order": np.zeros((layer.bias.shape)),
                        }

            n_samples = inputs.shape[0]
            inputs_shape = inputs.shape[1:]
            gradients = []
            bias_gradients = []
            for layer in model.layers:
                if layer.has_weights:
                    gradients.append(np.zeros(layer.d_weights.shape))
                if layer.has_bias:
                    bias_gradients.append(np.zeros(layer.d_bias.shape))
                # Iterate over batch
            for sample_idx in range(n_samples):
                # Compute forward pass on network, obtaining predictions and
                # setting internal variables required for backward pass.
                pred = model.forward(inputs[sample_idx].reshape(-1, *inputs_shape))
                # Compute loss derivative (error term for output layer)
                temp_t = self.loss.backward(pred, targets[sample_idx])
                # Backpropagate through network, accumulating gradients
                gradient_idx = -1
                bias_idx = -1
                for layer_idx in range(1, len(model.layers) + 1):
                    temp_t = model.layers[-layer_idx].backward(temp_t)
                    if model.layers[-layer_idx].has_weights:
                        gradients[gradient_idx] += model.layers[-layer_idx].d_weights
                        gradient_idx -= 1
                    if model.layers[-layer_idx].has_bias:
                        bias_gradients[bias_idx] += model.layers[-layer_idx].d_bias
                        bias_idx -= 1
            # Update weights by obtaining adam update term, averaging over batch
            # and multiplying by learning rate.
            gradient_idx = 0
            bias_idx = 0
            for i, layer in enumerate(model.layers):
                if layer.has_weights:
                    layer.weights -= (
                        self.learning_rate
                        * self.get_update(gradients[gradient_idx], i)
                        / n_samples
                    )
                    gradient_idx += 1
                if layer.has_bias:
                    layer.bias -= (
                        self.learning_rate
                        * self.get_update(bias_gradients[bias_idx], f"bias_{bias_idx}")
                        / n_samples
                    )
                    bias_idx += 1

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
        return (first_order) / (np.sqrt(second_order) + self.epsilon)
