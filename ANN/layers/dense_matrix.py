"""Dense layer implementation ommitting neuron objects
"""
import numpy as np
from numpy.typing import NDArray
from ANN.layers import Layer
from ANN.layers.initializers import gorlot
from ANN.activation_functions import Activation
from ANN.loss_functions import Loss
from ANN.optimizers import Optimizer

class DenseMatrix(Layer):

    def __init__(
            self,
            n_inputs : int,
            n_neurons : int,
            activation : Activation,
            loss: Loss
        ):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = gorlot(
            n_inputs,
            n_neurons,
            (n_inputs+1)*n_neurons
        ).reshape(n_inputs + 1, n_neurons)
        self.inputs = np.ones((1, n_inputs+1))
        self.gradients = np.zeros(self.weights.shape)
        self.activation_function = activation
        self.loss_function = loss

        def forward(inputs : NDArray[np.float32]) -> NDArray[np.float32]:
            if len(inputs.shape) != 2:
                raise ValueError(
                    f"Expected input array of shape (n_samples, n_inputs), \
                    but received {inputs.shape}."
                )
            outputs = np.zeros((inputs.shape[0], self.n_neurons))
            for i in range(inputs.shape[0]):
                self.inputs[:, :-1] = inputs[i]
                outputs[i] = self.activation_function.forward(np.dot(self.inputs, self.weights))
            return outputs

        def backward(
                inputs : NDArray[np.float32],
                targets : NDArray[np.float32],
                optimizer : Optimizer
            ):
            if len(inputs.shape) != 2:
                raise ValueError(
                    f"Expected input array of shape (n_samples, n_inputs), but received {inputs.shape}."
                )
            factor = 1 / inputs.shape[0]
            self.gradients = 0
            for i in range(inputs.shape[0]):

                self.inputs[:,:-1] = inputs[i]

                self.gradients += optimizer.get_update(
                    self,
                    targets[i]
                ) * factor

            self.weights -= self.gradients

            return np.sum(self.gradients, axis=0)

        Layer.__init__(self, forward, backward)
    