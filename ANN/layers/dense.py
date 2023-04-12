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

        # Set properties that have to be provided
        self.n_neurons = n_neurons
        self.activation_function = activation
        self.l1 = l1
        self.l2 = l2

        # Weights, bias and inputs may all be initialized later
        self.weights = None
        self.d_weights = None

        self.bias = None
        self.d_bias = None

        self.inputs = None

        # Mark layer as unitialized
        self.initialized = False

        if input_shape is not None:
            self.initialize((input_shape))

        Layer.__init__(
            self,
            has_weights=True,
            has_bias=True,
            input_shape=input_shape,
            output_shape=(1, n_neurons),
        )

    def forward(
        self, inputs: NDArray[cp.float32], training: bool = False
    ) -> NDArray[cp.float32]:
        """Compute forward pass

        Args:
            inputs (NDArray [cp.float32]): Input array of shape (n_samples, n_inputs)

        Returns:
            NDArray [cp.float32]: Activation values.
        """
        if len(inputs.shape) != 2:
            raise ShapeError(
                f"Expected input to have shape (n_samples, n_inputs)."
                " Received {inputs.shape=} instead."
            )
        # Initialize layer if necessary
        if self.initialized is False:
            self.initialize(inputs[0].shape)

        # Store inputs if training for later gradients calculation
        if training:
            self.inputs = inputs

        # Compute and return activations
        return self.activation_function.forward(
            cp.dot(inputs, self.weights) + self.bias
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
        # Get derivative of outputs with regards to z
        d_activation = self.activation_function.backward(gradient)

        # Get bias gradient
        self.d_bias = cp.sum(d_activation, axis=0)

        # Get weights gradient
        self.d_weights = cp.dot(self.inputs.T, d_activation)

        # Add regularization terms to gradient
        if self.l1 is not None:
            self.d_weights[self.weights > 0] += self.l1
            self.d_weights[self.weights < 0] -= self.l1
        if self.l2 is not None:
            self.d_weights -= self.l2 * self.weights
            self.d_bias -= self.l2 * self.bias

        # Get inputs gradient
        return cp.dot(d_activation, self.weights.T)

    def initialize(self, input_shape: Tuple[int, int]):
        # Store shape of input
        if len(input_shape) < 2:
            raise ShapeError(
                f"Expected input shape (n_samples, n_inputs). Got {input_shape=} instead."
            )
        # Set input shape
        self.input_shape = input_shape

        # Initialize memory and values for weights and biases.
        self.weights = gorlot(
            input_shape[1], self.n_neurons, (input_shape[1]) * self.n_neurons
        ).reshape(self.input_shape[1], self.n_neurons)
        self.bias = cp.zeros((self.n_neurons))

        # Initialize memory for gradients.
        self.d_weights = cp.zeros(self.weights.shape, dtype=cp.float32)
        self.d_bias = cp.zeros(self.bias.shape, dtype=cp.float32)

        # Initialize memory for inputs.
        self.inputs = cp.ones(input_shape, dtype=cp.float32)

        # Mark layer as initialized.
        self.initialized = True
