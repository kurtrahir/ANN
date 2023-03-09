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

        def forward(inputs: NDArray[np.float32]) -> NDArray[np.float32]:
            """Computes forward pass on the network

            Args:
                inputs (NDArray[np.float32]): Inputs to process. Expect shape (n_samples, *input_shape,)

            Returns:
                NDArray[np.float32]: Outputs
            """
            if self.initialized is False:
                initialize_weights(inputs[0].shape)
            processed_x = inputs
            for layer in self.layers:
                processed_x = layer.forward(processed_x)
            return processed_x

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
