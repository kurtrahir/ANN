"""Dense layer implementation
"""
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ANN.activation_functions.activation import Activation
from ANN.activation_functions.reLu import ReLu
from ANN.layers.initializers import gorlot
from ANN.layers.layer import Layer


class Dense(Layer):
    """Implementation of a densely connected layer"""

    def __init__(
        self,
        n_neurons: int,
        activation: Activation = ReLu(),
        input_shape: Optional[int] = None,
    ):
        """Create a dense layer

        Args:
            n_neurons (int): Number of neurons
            activation (Activation, optional): Activation function of layer. Defaults to ReLu().
            input_shape (Optional[Tuple[int,...]], optional): Shape of inputs to be passed to layer. Defaults to None.
        """

        def initialize_weights(input_shape: Tuple[int, int]):
            # Store shape of input
            self.input_shape = input_shape
            # Initialize Weights according to given input shape
            self.weights = gorlot(
                input_shape[1], n_neurons, (input_shape[1] + 1) * n_neurons
            ).reshape(input_shape[1] + 1, n_neurons)
            # Create matrix for inputs with added bias term
            self.inputs = np.ones((1, input_shape[1] + 1))
            # Create matrix for weights derivative
            self.d_weights = np.zeros(self.weights.shape)
            self.initialized = True

        # Number of neurons and activation function have to be provided.
        self.n_neurons = n_neurons
        self.activation_function = activation
        # Create matrix for activation values
        self.output_shape = (1, n_neurons)
        self.outputs = np.zeros(self.output_shape)

        # Input shape and weights have to be declared
        # but can be initialized later.
        self.input_shape = None
        self.d_weights = None
        self.weights = None
        self.inputs = None
        self.linear_combo = None
        self.initialized = False

        if input_shape is not None:
            initialize_weights((1, input_shape))

        def forward(inputs: NDArray[np.float32]) -> NDArray[np.float32]:
            """Compute forward pass

            Args:
                inputs (NDArray[np.float32]): Input array containing one sample.

            Returns:
                NDArray[np.float32]: Activation values.
            """
            if self.initialized is False:
                initialize_weights(inputs[0].shape)
            # Set input values (skipping bias at the end of the input array property)
            self.inputs[0, :-1] = inputs.reshape(1, -1)
            self.linear_combo = np.dot(self.inputs, self.weights)
            # Compute and store activations
            self.outputs = self.activation_function.forward(self.linear_combo)
            # Return activations
            return self.outputs

        def backward(error: NDArray[np.float32]) -> NDArray[np.float32]:
            """Compute backward pass

            Args:
                error (NDArray[np.float32]): Error matrix to propagate.

            Returns:
                NDArray[np.float32]: Input gradients for backpropagation.
            """
            # Get derivative of outputs with regards to dot product
            d_activation = self.activation_function.backward(self.linear_combo)
            # Get derivative of loss with regards to weights and store in
            # gradient property
            self.d_weights = np.dot(self.inputs.T, error * d_activation)
            # Get derivative of output with regards to inputs.
            return np.dot(error * d_activation, self.weights[:-1, :].T)

        Layer.__init__(
            self,
            forward=forward,
            backward=backward,
            initialize_weights=initialize_weights,
            has_weights=True,
            has_bias=False,
            weights=self.weights,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
        )
