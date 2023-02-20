"""Network implementation
"""
import numpy as np
from ANN import Layer


class Network:
    """Implementation of a neural network object
    """
    def __init__(self, layers : list[Layer]):
        self.layers = layers

    def add_layer(self, layer):
        """Add layer to network

        Args:
            layer (Layer): A layer object
        """
        self.layers.append(layer)

    def forward(self, inputs):
        """Computes forward pass on the network"""
        output = self.layers[0].forward(inputs).reshape(-1)
        for i in range(1, len(self.layers)):
            output = self.layers[i].forward(output).reshape(-1)
        return output

    def backward(self, inputs, targets, step_size):
        """Computes backward pass on the network"""
        outputs = [self.layers[0].forward(inputs).reshape(-1)]
        for i in range(1, len(self.layers)):
            outputs.append(self.layers[i].forward(outputs[-1]).reshape(-1))
        targets = self.layers[-1].backward(outputs[-2], targets, step_size).reshape(-1)
        for i in range(-2, -len(self.layers)):
            targets = self.layers[i].backward(outputs[i-1], targets, step_size).reshape(-1)

    def initialize_gorlot(self) -> None:
        """Initialize weights according to method proposed by gorlot
        """
        rnd = np.random.default_rng()
        def gorlot(n_inputs, n_neurons):
            value = np.sqrt(6) / np.sqrt(n_inputs + n_neurons)
            return rnd.uniform(-value, value, n_inputs)

        for layer in self.layers:
            layer.initialize_weights(gorlot)
