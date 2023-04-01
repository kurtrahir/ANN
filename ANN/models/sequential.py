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
        self.optimizer.backward(self, inputs, targets)

    def initialize_weights(self, input_shape):
        for layer in self.layers:
            layer.initialize(input_shape)
            input_shape = layer.output_shape
        self.initialized = True

    def save_model(self, path: str):
        model_dict = {}
        layer_name_dict = {}

        for layer in self.layers:
            layer_name = layer.__class__.__name__
            if layer_name not in layer_name_dict.keys():
                layer_name_dict[layer_name] = 0
            layer_name += f"_{layer_name_dict[layer_name]}"
            layer_name_dict[layer_name] += 1
            if layer.has_weights:
                model_dict[layer_name] = [layer.weights.get()]
            if layer.has_bias:
                model_dict[layer_name].append(layer.bias.get())

        with open(path, "wb") as f:
            pickle.dump(model_dict, f)
