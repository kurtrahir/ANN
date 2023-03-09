"""Convolutional layer implementation
"""

import operator
from functools import reduce
from typing import Literal, Optional, Tuple, Union

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ANN.activation_functions.activation import Activation
from ANN.activation_functions.reLu import ReLu
from ANN.correlate import corr2d_multi_in_out
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
        def initialize_weights(input_shape):
            self.input_shape = input_shape
            # Compute adjusted input shape
            self.inputs = pad(
                np.zeros(input_shape), self.kernel_shape, self.step_size, self.padding
            )
            self.padded_shape = self.inputs.shape

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

        def forward(inputs: NDArray[np.float32]):
            if len(inputs.shape) == 1:
                raise ShapeError(
                    f"Convolutional layer expected input of at least 2 dimensions, \
                    received {inputs.shape=} instead."
                )
            if len(inputs.shape) == 2:
                inputs = inputs.reshape(*inputs.shape, 1)

            # If first time seeing input shape, initiaze layer.
            if self.initialized is False:
                initialize_weights(inputs.shape)

            self.inputs = pad(inputs, self.kernel_shape, self.step_size, self.padding)

            self.lin_comb = (
                corr2d_multi_in_out(self.inputs, self.weights, self.step_size)
                + self.bias
            )

            self.outputs = self.activation_function.forward(self.lin_comb)

            return self.outputs

        def backward(error: NDArray[np.float32]):
            def dilate(
                array: NDArray[np.float32], step_size: Tuple[int, int]
            ) -> NDArray[np.float32]:
                output = np.zeros(
                    (
                        array.shape[0] + (step_size[0] - 1) * (array.shape[0] - 1),
                        array.shape[1] + (step_size[1] - 1) * (array.shape[1] - 1),
                        array.shape[2],
                    )
                )

                output[:: step_size[0], :: step_size[1]] = array

                return output

            # Get activation function gradient
            d_activation = self.activation_function.backward(self.lin_comb) * error

            # Get bias gradient
            self.d_bias += np.sum(d_activation * self.outputs, axis=(0, 1))

            # Dilate Error
            if self.step_size != (1, 1):
                d_activation = dilate(d_activation, self.step_size)

            # Rotate Kernel
            rot_weights = np.rot90(self.weights, 2, (1, 2))

            for c in range(self.d_weights.shape[-1]):
                self.d_weights[..., c] += np.swapaxes(
                    corr2d_multi_in_out(
                        self.inputs[:, :, c, np.newaxis],
                        np.swapaxes(d_activation, 2, 0)[..., np.newaxis],
                        (1, 1),
                    ),
                    2,
                    0,
                )

            if self.step_size != (1, 1):
                pad_w = self.kernel_shape[0] - 1
                pad_h = self.kernel_shape[1] - 1
                d_activation = np.pad(
                    d_activation, [[pad_w, pad_w], [pad_h, pad_h], [0, 0]], "constant"
                )
            else:
                d_activation = pad(
                    d_activation, self.kernel_shape, self.step_size, "full"
                )

            input_gradient = np.zeros(self.padded_shape)

            input_gradient = corr2d_multi_in_out(
                d_activation, np.swapaxes(rot_weights, 3, 0), (1, 1)
            )

            if self.padding == "valid":
                if self.padded_shape < self.input_shape:
                    dif_w = self.input_shape[0] - self.padded_shape[0]
                    dif_h = self.input_shape[1] - self.padded_shape[1]
                    input_gradient = np.pad(
                        input_gradient, [[0, dif_w], [0, dif_h], [0, 0]], "constant"
                    )

            return input_gradient

        def pad(
            input_a: NDArray[np.float32],
            shape_b: Tuple[int, ...],
            step_size: Tuple[int, int],
            padding: Literal["valid", "full", "same"],
        ) -> NDArray[np.float32]:
            """Pad array to match desired convolution procedure when convolved with kernel of specified shape

            Args:
                input_a (NDArray[np.float32]): Array to be padded
                shape_b (Tuple[int,...]): Shape array will be convolved with
                step_size (Tuple[int, int]): Step size to be used in convolution
                padding ( Literal["valid", "full", "same"]): Padding strategy to employ

            Returns:
                NDArray[np.float32]: Padded array
            """

            def get_max_valid_idx(size_a, size_b, stride):
                res = (size_a - size_b) % stride
                return -res if res != 0 else size_a

            def get_size(width, kernel_size, stride):
                return np.ceil((stride * (width - 1) - width + kernel_size) / 2).astype(
                    int
                )

            pad_size = (0, 0)
            if padding == "same":
                pad_size = (
                    get_size(input_a.shape[0], shape_b[0], step_size[0]),
                    get_size(input_a.shape[1], shape_b[1], step_size[1]),
                )
            elif padding == "full":
                pad_size = (shape_b[0] - 1, shape_b[1] - 1)
            elif padding == "valid":
                return input_a[
                    : get_max_valid_idx(input_a.shape[0], shape_b[0], step_size[0]),
                    : get_max_valid_idx(input_a.shape[1], shape_b[1], step_size[1]),
                    :,
                ]
            padded_input = np.pad(
                input_a,
                [[pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [0, 0]],
                "constant",
            )
            return padded_input

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

        # verify valid setup
        if step_size > kernel_shape:
            raise ShapeError(
                f"Step size larger than kernel size will result in omitted inputs. \
                {step_size=}, {kernel_shape=}"
            )

        # pass input_shape through -> it can be None
        if input_shape is not None:
            if input_shape[:-1] < kernel_shape:
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
