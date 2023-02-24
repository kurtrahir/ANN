"""Adam optimizer"""

import numpy as np
from numpy.typing import NDArray
from ANN.optimizers import Optimizer


class Adam(Optimizer):

    def __init__(self, beta_1 = 0.9, beta_2 = 0.999, learning_rate = 1e-3, epsilon = 1e-15):

        self.momentums = {}
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        def backward(
            model,
            inputs : NDArray[np.float32],
            targets: NDArray[np.float32]
        ):
            if not self.momentums:
                for i,layer in enumerate(model.layers):
                    self.momentums[i] = {
                        "first_order" : np.zeros((layer.weights.shape)),
                        "second_order" : np.zeros((layer.weights.shape))
                    }

            n_samples = inputs.shape[0]
            n_inputs = inputs.shape[1]
            n_layers = len(model.layers)
            gradients =  [
                    np.zeros(layer.weights.shape) for layer in model.layers
                ]

            for i in range(n_samples):
                outputs = model.layers[0].forward(
                    inputs[i].reshape(-1, n_inputs))
                for i in range(1, len(model.layers)):
                    outputs = model.layers[i].forward(outputs)

                gradients[-1] += self.get_update(
                    model.layers[-1].backward(targets),
                    n_layers-1
                )
                for i in range(2, len(model.layers)):
                    gradients[-i] += self.get_update(
                        model.layers[-i].backward(
                            gradients[-i+1]
                        ),
                        n_layers-i
                    )
                
                for i, layer in enumerate(model.layers):
                    layer.weights -= self.learning_rate * gradients[i] / n_samples

        Optimizer.__init__(
            self,
            backward=backward
        )

    def update_momentums(self, gradients, layer_idx):
        """Update first order and second order momentums"""
        self.momentums[layer_idx]["first_order"] *= self.beta_1 
        self.momentums[layer_idx]["first_order"] += (1-self.beta_2) * gradients
        self.momentums[layer_idx]["second_order"] *= self.beta_2 
        self.momentums[layer_idx]["second_order"] += (1-self.beta_2) * (gradients**2)

    def get_update(self, gradients, layer_idx):
        """Compute update term"""
        self.update_momentums(gradients, layer_idx)
        first_order = self.momentums[layer_idx]["first_order"] / (1-self.beta_1)
        second_order = self.momentums[layer_idx]["second_order"] / \
            (1-self.beta_2)
        return (first_order) / (np.sqrt(second_order) + self.epsilon)
