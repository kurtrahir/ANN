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
        self.initialized = False

        def initialize_weights(input_shape):
            for layer in self.layers:
                layer.initialize_weights(input_shape)
                input_shape = layer.output_shape
            self.initialized = True

        def forward(inputs: NDArray[np.float32]):
            """Computes forward pass on the network"""

            def one_forward(one_x):
                """Pass one sample through the model"""
                for layer in self.layers:
                    one_x = layer.forward(one_x)
                return one_x

            if self.initialized is False:
                initialize_weights(inputs[0].shape)

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

        Model.__init__(
            self,
            forward=forward,
            backward=backward,
            layers=layers,
            optimizer=optimizer,
            initialize_weights=initialize_weights,
        )

    def add_layer(self, layer):
        """Add layer to network

        Args:
            layer (Layer): A layer object
        """
        self.layers.append(layer)
