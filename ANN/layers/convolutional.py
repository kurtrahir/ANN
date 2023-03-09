"""Convolutional layer implementation
"""

import operator
from functools import reduce
from typing import Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ANN.activation_functions.activation import Activation
from ANN.activation_functions.reLu import ReLu
from ANN.correlate.correlate import corr2d_multi_in_out
from ANN.correlate.pad import pad
from ANN.errors.shapeError import ShapeError
from ANN.layers.initializers import gorlot
from ANN.layers.layer import Layer


class Conv2D(Layer):
    def __init__(
        self,
        n_filters: int,
        kernel_shape: Union[int, tuple[int, int]],
        step_size: Union[int, Tuple[int, int]] = (1, 1),
        input_shape: Optional[Tuple[int, int, int]] = None,
        activation_function: Activation = ReLu(),
        padding: Optional[Literal["full", "valid", "same"]] = "full",
    ):
        # verify valid setup
        if step_size > kernel_shape:
            raise ShapeError(
                f"Step size larger than kernel size will result in omitted inputs. \
                {step_size=}, {kernel_shape=}"
            )

        def initialize_weights(input_shape: Tuple[int, int, int]):
            """Initialize the weights for the layer using provided input shape.

            Args:
                input_shape (Tuple[int, int, int]): Input shape (x_dim, y_dim, channels)
            """
            self.input_shape = input_shape
            # Compute adjusted input shape
            self.inputs = pad(
                np.zeros((1,) + input_shape),
                self.kernel_shape,
                self.step_size,
                self.padding,
            )
            self.padded_shape = self.inputs.shape[1:]

            # Compute output shape
            self.output_shape = (
                *get_shape(self.padded_shape, self.kernel_shape, step_size),
                n_filters,
            )

            self.outputs = np.zeros(self.output_shape)

            # Initialize weights and gradients
            weights_shape = (n_filters, *self.kernel_shape, self.padded_shape[2])
            n_inputs = reduce(operator.mul, self.padded_shape, 1)
            n_outputs = reduce(operator.mul, weights_shape, 1)
            self.weights = gorlot(n_inputs, n_outputs, weights_shape)
            self.d_weights = np.zeros(weights_shape)
            self.bias = gorlot(n_inputs, n_outputs, (n_filters,))
            self.d_bias = np.zeros(self.bias.shape)

            self.initialized = True

        def forward(inputs: NDArray[np.float32]) -> NDArray:
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
                initialize_weights(inputs.shape[1:])

            # Set input values for access by backward())
            self.inputs = pad(inputs, self.kernel_shape, self.step_size, self.padding)

            # Set z value for access by backward()
            self.lin_comb = (
                corr2d_multi_in_out(self.inputs, self.weights, self.step_size)
                + self.bias
            )
            # Set activations for use by backward()
            self.outputs = self.activation_function.forward(self.lin_comb)

            return self.outputs

        def backward(error: NDArray[np.float32]) -> NDArray[np.float32]:
            """Compute backward pass using provided error term.

            Args:
                error (NDArray[np.float32]): Error should match self.outputs shape.

            Returns:
                NDArray[np.float32]: Returns loss gradient with regards to inputs for backward
                propagation.
            """

            def dilate(
                array: NDArray[np.float32], step_size: Tuple[int, int]
            ) -> NDArray[np.float32]:
                """Dilate array using provided step size

                Args:
                    array (NDArray[np.float32]): Array to dilate, expects (n_samples, x_dim, y_dim,...)
                    step_size (Tuple[int, int]): Step size used in forward convolution
                    to decide the appropriate dilation size.

                Returns:
                    NDArray[np.float32]: Dilated Array
                """
                output = np.zeros(
                    (
                        array.shape[0],
                        array.shape[1] + (step_size[0] - 1) * (array.shape[1] - 1),
                        array.shape[2] + (step_size[1] - 1) * (array.shape[2] - 1),
                        *array.shape[3:],
                    )
                )

                output[:, :: step_size[0], :: step_size[1]] = array

                return output

            # Get activation function gradient
            d_activation = self.activation_function.backward(self.lin_comb) * error

            # Get bias gradient
            self.d_bias += np.sum(d_activation * self.outputs, axis=(0, 1, 2))

            # Dilate Error
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
                d_activation = pad(
                    d_activation, self.kernel_shape, self.step_size, "full"
                )

            input_gradient = corr2d_multi_in_out(
                d_activation, np.swapaxes(rot_weights, 3, 0), (1, 1)
            )

            if self.padding == "valid":
                if self.padded_shape < self.input_shape:
                    dif_w = self.input_shape[0] - self.padded_shape[0]
                    dif_h = self.input_shape[1] - self.padded_shape[1]
                    input_gradient = np.pad(
                        input_gradient,
                        [[0, 0], [0, dif_w], [0, dif_h], [0, 0]],
                        "constant",
                    )

            return input_gradient

        # utility function for shorthand in the case of a square filter (3) -> (3,3)
        def unpack(tuple_or_int: Union[Tuple[int, int], int]) -> Tuple[int, int]:
            return (
                (tuple_or_int, tuple_or_int)
                if isinstance(tuple_or_int, int)
                else tuple_or_int
            )

        # Get shape that would result from convolving a with b using provided step size.
        def get_shape(shape_a, shape_b, step_size):
            return (
                np.ceil((shape_a[0] - shape_b[0]) / step_size[0]).astype(int) + 1,
                np.ceil((shape_a[1] - shape_b[1]) / step_size[1]).astype(int) + 1,
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
        self.outputs = None
        self.initialized = False
        self.lin_comb = None

        # pass input_shape through -> it can be None
        if input_shape is not None:
            if input_shape[1:-1] < kernel_shape:
                raise ShapeError(
                    f"Input size is smaller than kernel size. {input_shape=}, {kernel_shape=}"
                )
            # If not none and valid shape, initialize layer
            initialize_weights(input_shape)

        Layer.__init__(
            self,
            forward=forward,
            backward=backward,
            initialize_weights=initialize_weights,
            has_weights=True,
            has_bias=True,
            weights=self.weights,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
        )
