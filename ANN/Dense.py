"""Fully Connected Layer
"""

from Layer import Layer
from Activation import Activation
from Loss import Loss
from Neuron import Neuron
import numpy as np

class Dense(Layer):
    """Implementation of a fully connected layer"""

    def __init__(self, n_inputs : int, n_neurons : int, activation : Activation, loss : Loss):
        self.Neurons = np.array(
            [
                Neuron(
                    n_inputs = n_inputs,
                    activation = activation,
                    loss = loss
                )
                for i in range(n_neurons)
            ]
        )

        def forward(inputs):
            return np.array(
                [
                    n.forward(inputs) for n in self.Neurons
                ]
            )


        def backward(inputs, target, step_size):
            for i, neuron in enumerate(self.Neurons):
                neuron.backward(
                    inputs,
                    target[i],
                    step_size
                )


        Layer.__init__(
            self,
            forward=forward,
            backward=backward
        )
