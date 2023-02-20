"""Fully Connected Layer
"""
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from ANN import Neuron
from ANN.layers import Layer
from ANN.loss_functions import Loss
from ANN.activation_functions import Activation

class Dense(Layer):
    """Implementation of a fully connected layer"""

    def __init__(self, n_inputs : int, n_neurons : int, activation : Activation, loss : Loss):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.Neurons = np.array(
            [
                Neuron(
                    n_inputs = n_inputs,
                    activation_function = activation,
                    loss_function = loss
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
            gradients = []
            for i, neuron in enumerate(self.Neurons):
                gradients.append(
                    neuron.backward(
                        inputs,
                        target[i],
                        step_size
                    )
                )
            return np.array(gradients)

        Layer.__init__(
            self,
            forward=forward,
            backward=backward
        )

    def initialize_weights(
        self,
        initializer : Callable[[int,int], NDArray[np.float32]]
    ) -> None:
        for neuron in self.Neurons:
            neuron.set_weights(initializer(self.n_inputs, self.n_neurons, self.n_inputs+1))
