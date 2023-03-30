"""Dense layer implementation
"""
from typing import Optional, Tuple

import cupy as np
from cupy.typing import NDArray

from ANN.activation_functions.activation import Activation
from ANN.activation_functions.reLu import ReLu
from ANN.errors.shapeError import ShapeError
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

        # Number of neurons and activation function have to be provided.
        self.n_neurons = n_neurons
        self.activation_function = activation
        # Create matrix for activation values
        self.output_shape = (1, n_neurons)

        # Input shape and weights have to be declared
        # but can be initialized later.
        self.input_shape = None
        self.d_weights = None
        self.weights = None
        self.bias = None
        self.d_bias = None
        self.inputs = None
        self.initialized = False

        if input_shape is not None:
            self.initialize((input_shape))

        Layer.__init__(
            self,
            has_weights=True,
            has_bias=True,
            weights=self.weights,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
        )

    def forward(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute forward pass

        Args:
            inputs (NDArray[np.float32]): Input array of shape (n_samples, n_inputs)

        Returns:
            NDArray[np.float32]: Activation values.
        """
        if len(inputs.shape) != 2:
            raise ShapeError(
                f"Expected input to have shape (n_samples, n_inputs). \
                Received {inputs.shape=} instead."
            )

        if self.initialized is False:
            self.initialize(inputs[0].shape)
        # Set input values (skipping bias at the end of the input array property)
        self.inputs = inputs
        # Compute and return activations
        return self.activation_function.forward(
            np.dot(self.inputs, self.weights) + self.bias
        )

    def backward(self, gradient: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute backward pass

        Args:
            error (NDArray[np.float32]): Error matrix to propagate.
            Expect shape (n_samples, activations)

        Returns:
            NDArray[np.float32]: Input gradients for backpropagation.
            Shape (n_samples, input gradients)
        """
        # Get derivative of outputs with regards to dot product
        d_activation = self.activation_function.backward(gradient)
        # Get derivative of loss with regards to weights and store in
        # gradient property
        self.d_bias += np.sum(d_activation, axis=0)
        self.d_weights += np.dot(self.inputs.T, d_activation)
        # Get derivative of output with regards to inputs.
        return np.dot(d_activation, self.weights.T)

    def initialize(self, input_shape: Tuple[int, int]):
        # Store shape of input
        if len(input_shape) < 2:
            raise ShapeError(
                f"Expected input shape (n_samples, n_inputs). Got {input_shape=} instead."
            )
        self.input_shape = input_shape
        # Initialize Weights according to given input shape
        self.weights = gorlot(
            input_shape[1], self.n_neurons, (input_shape[1]) * self.n_neurons
        ).reshape(self.input_shape[1], self.n_neurons)
        self.bias = np.zeros((self.n_neurons))
        # Create matrix for inputs with added bias term
        self.inputs = np.ones(input_shape, dtype=np.float32)
        # Create matrix for weights derivative
        self.d_weights = np.zeros(self.weights.shape, dtype=np.float32)
        self.d_bias = np.zeros(self.bias.shape, dtype=np.float32)
        self.initialized = True
