"""Dense layer implementation ommitting neuron objects
"""
import numpy as np
from numpy.typing import NDArray
from ANN.layers import Layer
from ANN.layers.initializers import gorlot
from ANN.activation_functions import Activation
from ANN.loss_functions import Loss

class DenseMatrix(Layer):

    def __init__(
            self,
            n_inputs : int,
            n_neurons : int,
            activation : Activation,
            loss: Loss
        ):
        # Store shape of layer
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        # Initialize Weights
        self.weights = gorlot(
            n_inputs,
            n_neurons,
            (n_inputs+1)*n_neurons
        ).reshape(n_inputs + 1, n_neurons)
        # Create matrix for inputs with added bias term
        self.inputs = np.ones((1, n_inputs+1))
        # Create matrix for activation values
        self.outputs = np.zeros((1, n_neurons))
        # Create matrix for weights derivative
        self.d_weights = np.zeros(self.weights.shape)
        # Create matrix for inputs derivative
        self.d_inputs = np.zeros((1,n_inputs))
        # Set activation and loss function
        self.activation_function = activation
        self.loss_function = loss
        self.linear_combination = np.zeros(np.dot(self.inputs, self.weights).shape)

        def forward(inputs : NDArray[np.float32]) -> NDArray[np.float32]:
            self.inputs[:, :-1] = inputs
            self.linear_combination = np.dot(self.inputs, self.weights)
            self.outputs = self.activation_function.forward(self.linear_combination)
            return self.outputs

        def backward(
            targets: NDArray[np.float32]
        ):
            # Get loss derivative with regards to output.
            d_loss = self.loss_function.backward(self.outputs, targets)
            # Get derivative of outputs with regards to dot product
            d_activation = self.activation_function.backward(self.linear_combination)
            # Derivative of dot product with regards to weights is the inputs.
            # Get derivative of loss with regards to weights:
            self.d_weights = np.dot(self.inputs.T, d_loss * d_activation)

            # Get derivative of output with regards to inputs.
            self.d_inputs = np.dot(d_loss * d_activation, self.weights[:-1,:].T)

            return self.d_inputs

        Layer.__init__(self, forward, backward, self.d_weights)
