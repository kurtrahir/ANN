"""Network implementation
"""

import numpy as np
from numpy.typing import NDArray

from ANN.layers import Layer
from ANN.models import Model
from ANN.optimizers import Optimizer


class Sequential(Model):
    """Implementation of a neural network object"""

    def __init__(self, layers: list[Layer], optimizer: Optimizer):
        def forward(inputs):
            """Computes forward pass on the network"""
            output = np.zeros((inputs.shape[0], self.layers[-1].n_neurons))
            # Iterate over batch
            for i in range(inputs.shape[0]):
                # Compute forward pass by passing previous output as input
                # to next layer
                layer_output = self.layers[0].forward(inputs[i])
                for j in range(1, len(self.layers)):
                    layer_output = self.layers[j].forward(layer_output)
                # Store final output (output layer)
                output[i] = layer_output
            return output

        def backward(inputs: NDArray[np.float32], targets: NDArray[np.float32]):
            """Computes backward pass on the network"""
            # Hand off backward pass to optimizer
            self.optimizer.backward(self, inputs, targets)

        Model.__init__(self, forward, backward, layers, optimizer)

    def add_layer(self, layer):
        """Add layer to network

        Args:
            layer (Layer): A layer object
        """
        self.layers.append(layer)
