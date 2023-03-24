"""Convolutional layer implementation
"""

import operator
from functools import reduce
from typing import Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ANN.activation_functions.activation import Activation
from ANN.correlate.correlate import corr2d_multi_in_out
from ANN.correlate.dilate import dilate
from ANN.correlate.pad import pad, unpad
from ANN.correlate.shape import get_shape
from ANN.errors.shapeError import ShapeError
from ANN.layers.initializers import gorlot
from ANN.layers.layer import Layer


class Conv2D(Layer):
    def initialize(self, input_shape: Tuple[int, int, int, int]):
        """Initialize the weights for the layer using provided input shape.

        Args:
            input_shape (Tuple[int, int, int, int]): Input shape (n_samples, x_dim, y_dim, channels)
        """
        if len(input_shape) < 4:
            raise ShapeError(
                f"Expected input shape (n_samples, x_dim, y_dim, channels). Got {input_shape=} instead."
            )
        if input_shape[1:3] < self.kernel_shape:
            raise ShapeError(
                f"Input size is smaller than kernel size. {input_shape=}, {self.kernel_shape=}"
            )
        self.input_shape = input_shape
        # Compute adjusted input shape
        self.inputs = pad(
            np.zeros(input_shape),
            self.kernel_shape,
            self.step_size,
            self.padding,
        )
        self.padded_shape = self.inputs.shape

        # Compute output shape
        self.output_shape = get_shape(
            self.padded_shape,
            (self.n_filters,) + self.kernel_shape + (self.input_shape[-1],),
            self.step_size,
        )
        # Initialize weights and gradients

        # Find shape of weights
        weights_shape = (self.n_filters, *self.kernel_shape, self.padded_shape[-1])
        # Calculate number of inputs and outputs for gorlot initialization
        n_inputs = reduce(operator.mul, self.padded_shape[1:], 1)
        n_outputs = reduce(operator.mul, weights_shape, 1)
        # Initialize weights and weight gradients
        self.weights = gorlot(n_inputs, n_outputs, weights_shape)
        self.d_weights = np.zeros(weights_shape)
        # Initialize bias and bias gradients.
        self.bias = gorlot(n_inputs, n_outputs, (self.n_filters,))
        self.d_bias = np.zeros(self.bias.shape)
        # Set initialization marker
        self.initialized = True

    def __init__(
        self,
        n_filters: int,
        kernel_shape: Union[int, tuple[int, int]],
        activation_function: Activation,
        step_size: Union[int, Tuple[int, int]] = (1, 1),
        input_shape: Optional[Tuple[int, int, int]] = None,
        padding: Optional[Literal["full", "valid", "same"]] = "full",
    ):
        # verify valid setup
        if step_size > kernel_shape:
            raise ShapeError(
                f"Step size larger than kernel size will result in omitted inputs. \
                {step_size=}, {kernel_shape=}"
            )

        # utility function for shorthand in the case of a square filter (3) -> (3,3)
        def unpack(tuple_or_int: Union[Tuple[int, int], int]) -> Tuple[int, int]:
            return (
                (tuple_or_int, tuple_or_int)
                if isinstance(tuple_or_int, int)
                else tuple_or_int
            )

        # n_filters, kernel shape, step size, padding and activation function
        # are always set on init
        self.n_filters = n_filters
        self.kernel_shape = unpack(kernel_shape)
        self.step_size = unpack(step_size)
        self.padding = padding
        self.activation_function = activation_function

        # Declare input dependent variables
        self.input_shape = input_shape
        self.inputs = None
        self.padded_shape = None
        self.output_shape = None
        self.weights = None
        self.d_weights = None
        self.bias = None
        self.d_bias = None
        self.initialized = False

        # pass input_shape through -> it can be None
        if input_shape is not None:
            # If not none and valid shape, initialize layer
            self.initialize(input_shape)

        Layer.__init__(
            self,
            has_weights=True,
            has_bias=True,
            weights=self.weights,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
        )

    def forward(self, inputs: NDArray[np.float32]) -> NDArray:
        """Computes forward pass on the inputs provided.

        Args:
            inputs (NDArray[np.float32]): Inputs should have shape (n_samples, x_dim, y_dim, channels)

        Raises:
            ShapeError: If incorrectly shaped inputs are provided.

        Returns:
            NDArray[np.float32]: Outputs in shape (n_samples, x_dim, y_dim, feature maps)
        """
        if len(inputs.shape) != 4:
            raise ShapeError(
                f"Convolutional layer expected input of shape (n_samples, x_dim, y_dim, channels), \
                received {inputs.shape=} instead."
            )

        # If first time seeing input shape, initiaze layer.
        if self.initialized is False:
            self.initialize(inputs.shape)

        # Set input values for access by backward())
        self.inputs = pad(inputs, self.kernel_shape, self.step_size, self.padding)

        # Convolve, add bias and return.
        return self.activation_function.forward(
            corr2d_multi_in_out(self.inputs, self.weights, self.step_size) + self.bias
        )

    def backward(self, gradient: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute backward pass using provided error term.

        Args:
            error (NDArray[np.float32]): Error should match self.outputs shape.

        Returns:
            NDArray[np.float32]: Returns loss gradient with regards to inputs for backward
            propagation.
        """
        # Get activation function gradient
        d_activation = self.activation_function.backward(gradient)

        # Get bias gradient
        self.d_bias += np.sum(
            d_activation * self.activation_function.activations, axis=(0, 1, 2)
        )

        # Dilate activation gradient
        if self.step_size != (1, 1):
            d_activation = dilate(d_activation, self.step_size)

        # Rotate Kernel
        rot_weights = np.rot90(self.weights, 2, (1, 2))

        for channel in range(self.d_weights.shape[-1]):
            self.d_weights[..., channel] += np.swapaxes(
                corr2d_multi_in_out(
                    np.swapaxes(self.inputs, 3, 0)[np.newaxis, channel],
                    np.swapaxes(d_activation, 3, 0),
                    (1, 1),
                ),
                3,
                0,
            )[..., 0]

        if self.step_size != (1, 1):
            pad_w = self.kernel_shape[0] - 1
            pad_h = self.kernel_shape[1] - 1
            d_activation = np.pad(
                d_activation,
                [[0, 0], [pad_w, pad_w], [pad_h, pad_h], [0, 0]],
                "constant",
            )
        else:
            d_activation = pad(d_activation, self.kernel_shape, self.step_size, "full")

        input_gradient = corr2d_multi_in_out(
            d_activation, np.swapaxes(rot_weights, 3, 0), (1, 1)
        )

        return unpad(input_gradient, self.kernel_shape, self.input_shape, self.padding)
