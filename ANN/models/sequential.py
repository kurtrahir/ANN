"""Network implementation
"""

import pickle

import cupy as cp
from cupy.typing import NDArray

from ANN.layers import Layer
from ANN.models import Model
from ANN.optimizers import Optimizer


class Sequential(Model):
    """Implementation of a neural network object"""

    def __init__(self, layers: list[Layer], optimizer: Optimizer):
        Model.__init__(
            self,
            layers=layers,
            optimizer=optimizer,
        )

    def forward(
        self, inputs: NDArray[cp.float32], training: bool = False
    ) -> NDArray[cp.float32]:
        """Computes forward pass on the network

        Args:
            inputs (NDArray [cp.float32]): Inputs to process. Expect shape (n_samples, *input_shape,)
            training (bool): indicates whether this is part of model training (controls behaviour of certain layers)
        Returns:
            NDArray [cp.float32]: Outputs
        """
        if self.initialized is False:
            self.initialize_weights(inputs.shape)
        processed_x = inputs
        for layer in self.layers:
            processed_x = layer.forward(processed_x, training=training)
        return processed_x

    def backward(self, inputs: NDArray[cp.float32], targets: NDArray[cp.float32]):
        """Computes backward pass on the network"""
        # Hand off backward pass to optimizer
        return self.optimizer.backward(self, inputs, targets)

    def initialize_weights(self, input_shape):
        for layer in self.layers:
            layer.initialize(input_shape)
            input_shape = layer.output_shape
        self.initialized = True
