"""Fully Connected Layer
"""

import numpy as np
from ANN import Neuron
from ANN.layers import Layer
from ANN.loss_functions import Loss
from ANN.activation_functions import Activation

class Dense(Layer):
    """Implementation of a fully connected layer"""

    def __init__(self, n_inputs : int, n_neurons : int, activation : Activation, loss : Loss):
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
