"""Dense layer implementation
"""
from typing import Optional, Tuple

import cupy as cp
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
        l1: Optional[cp.float32] = None,
        l2: Optional[cp.float32] = None,
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

        self.l1 = l1
        self.l2 = l2

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

    def forward(self, inputs: NDArray[cp.float32]) -> NDArray[cp.float32]:
        """Compute forward pass

        Args:
            inputs (NDArray [cp.float32]): Input array of shape (n_samples, n_inputs)

        Returns:
            NDArray [cp.float32]: Activation values.
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
            cp.dot(self.inputs, self.weights) + self.bias
        )

    def backward(self, gradient: NDArray[cp.float32]) -> NDArray[cp.float32]:
        """Compute backward pass

        Args:
            error (NDArray [cp.float32]): Error matrix to propagate.
            Expect shape (n_samples, activations)

        Returns:
            NDArray [cp.float32]: Input gradients for backpropagation.
            Shape (n_samples, input gradients)
        """
        # Get derivative of outputs with regards to dot product
        d_activation = self.activation_function.backward(gradient)
        # Get derivative of loss with regards to weights and store in
        # gradient property
        self.d_bias = cp.sum(d_activation, axis=0)
        self.d_weights = cp.dot(self.inputs.T, d_activation)
        if self.l1 is not None:
            self.d_weights[self.weights > 0] += self.l1
            self.d_weights[self.weights < 0] -= self.l1
        if self.l2 is not None:
            self.d_weights -= self.l2 * self.weights
            self.d_bias -= self.l2 * self.bias
        # Get derivative of output with regards to inputs.
        return cp.dot(d_activation, self.weights.T)

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
        self.bias = cp.zeros((self.n_neurons))
        # Create matrix for inputs with added bias term
        self.inputs = cp.ones(input_shape, dtype=cp.float32)
        # Create matrix for weights derivative
        self.d_weights = cp.zeros(self.weights.shape, dtype=cp.float32)
        self.d_bias = cp.zeros(self.bias.shape, dtype=cp.float32)
        self.initialized = True
