"""Network implementation
"""

import numpy as np
from numpy.typing import NDArray
from ANN.layers import Layer
from ANN.optimizers import Optimizer, SGD
from ANN import Network, gorlot

class Sequential(Network):
    """Implementation of a neural network object
    """
    def __init__(self, layers : list[Layer], gorlot_init : bool = True, optimizer : Optimizer = SGD(0.1)):
        self.layers = layers
        if gorlot_init:
            self.initialize_gorlot()

        def forward(self, inputs):
            """Computes forward pass on the network"""
            output = self.layers[0].forward(inputs)
            for i in range(1, len(self.layers)):
                output = self.layers[i].forward(output)
            return output

        def backward(
                self,
                inputs: NDArray[np.float32],
                targets: NDArray[np.float32],
                optimizer: Optimizer):
            """Computes backward pass on the network"""
            outputs = [self.layers[0].forward(inputs)]
            for i in range(1, len(self.layers)):
                outputs.append(self.layers[i].forward(outputs[-1]))
            targets = self.layers[-1].backward(
                outputs[-2],
                targets,
                optimizer
            )
            for i in range(-2, -len(self.layers)):
                targets = self.layers[i].backward(
                    outputs[i-1],
                    targets,
                    optimizer
                )
        Network.__init__(
            self,
            forward,
            backward,
            optimizer
        )

    def add_layer(self, layer):
        """Add layer to network

        Args:
            layer (Layer): A layer object
        """
        self.layers.append(layer)

    

    def initialize_gorlot(self) -> None:
        """Initialize weights according to method proposed by gorlot
        """
        for layer in self.layers:
            layer.initialize_weights(gorlot)
