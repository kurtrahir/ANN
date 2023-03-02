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

            def one_forward(one_input):
                """Pass one sample through the model"""
                layer_output = self.layers[0].forward(one_input)
                for j in range(1, len(self.layers)):
                    layer_output = self.layers[j].forward(layer_output)
                return layer_output

            # Get dummt output for shape
            dummy_output = one_forward(inputs[0])
            output = np.zeros((inputs.shape[0], *dummy_output.shape[1:]))
            output[0] = dummy_output
            # Iterate over batch
            for i in range(1, inputs.shape[0]):
                output[i] = one_forward(inputs[i])
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
