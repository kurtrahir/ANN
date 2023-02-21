"""Network implementation
"""
from ANN import Layer, gorlot

class Sequential:
    """Implementation of a neural network object
    """
    def __init__(self, layers : list[Layer], gorlot_init : bool = True):
        self.layers = layers
        self.training_steps = 0
        if gorlot_init:
            self.initialize_gorlot()

    def add_layer(self, layer):
        """Add layer to network

        Args:
            layer (Layer): A layer object
        """
        self.layers.append(layer)

    def forward(self, inputs):
        """Computes forward pass on the network"""
        output = self.layers[0].forward(inputs)
        for i in range(1, len(self.layers)):
            output = self.layers[i].forward(output)
        return output

    def backward(self, inputs, targets, step_size):
        """Computes backward pass on the network"""
        outputs = [self.layers[0].forward(inputs)]
        for i in range(1, len(self.layers)):
            outputs.append(self.layers[i].forward(outputs[-1]))
        targets = self.layers[-1].backward(outputs[-2], targets, step_size)
        for i in range(-2, -len(self.layers)):
            targets = self.layers[i].backward(outputs[i-1], targets, step_size)
        self.training_steps += 1

    def initialize_gorlot(self) -> None:
        """Initialize weights according to method proposed by gorlot
        """
        for layer in self.layers:
            layer.initialize_weights(gorlot)
