"""Convolutional layer implementation
"""

from typing import Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ANN.activation_functions.activation import Activation
from ANN.errors.shapeError import ShapeError
from ANN.layers.layer import Layer


class Conv2D(Layer):
    def __init__(
        self,
        input_size: Tuple[int, int, int],
        n_filters: int,
        kernel_size: Union[int, tuple[int, int]],
        activation_function: Activation,
        step_size: Union[int, Tuple[int, int]],
        padding: Optional[Literal["full", "valid", "same"]] = "full",
    ):
        def forward(inputs: NDArray[np.float32]):
            if len(inputs.shape) == 1:
                raise ShapeError(
                    f"Convolutional layer expected input of at least 2 dimensions, \
                    received {inputs.shape=} instead."
                )
            if len(inputs.shape) == 2:
                self.inputs = pad(
                    inputs.reshape(*inputs.shape, 1),
                    self.weights,
                    self.step_size,
                    self.padding,
                )
            else:
                self.inputs = pad(inputs, self.weights, self.step_size, self.padding)

            outputs = np.zeros(self.output_shape)
            for kernel in range(self.n_filters):
                for channel in range(self.input_size[2]):
                    outputs[:, :, kernel] += corr2d(
                        self.inputs[:, :, channel],
                        self.weights[:, :, kernel],
                        self.step_size,
                    )
            return self.activation_function.forward(outputs)

        def backward(error: NDArray[np.float32]):
            for kernel in range(self.d_weights.shape[2]):
                self.d_weights[:, :, kernel] = corr2d(
                    self.inputs[:, :, kernel],
                    error[:, :, kernel],
                    error.shape[:-1] if self.step_size != (1, 1) else (1, 1),
                )

            rot_weights = np.rot90(self.weights, 2, (0, 1))

            padded_error = pad(error, rot_weights, (1, 1), "full")

            new_error = np.zeros(
                (
                    *get_shape(
                        padded_error.shape[:-1], self.weights.shape[:-1], (1, 1)
                    ),
                    self.n_filters,
                )
            )
            for kernel in range(self.n_filters):
                new_error[:, :, kernel] = corr2d(
                    padded_error[:, :, kernel], rot_weights[:, :, kernel]
                )
            return new_error

        def corr2d(
            input_a: NDArray[np.float32],
            input_b: NDArray[np.float32],
            step_size: Tuple[int, int] = (1, 1),
        ):
            b_h, b_w = input_b.shape
            output_shape = get_shape(input_a.shape, input_b.shape, step_size)
            output = np.zeros(output_shape)
            for i in range(0, output_shape[0], step_size[0]):
                for j in range(0, output_shape[1], step_size[1]):
                    output[i, j] = np.sum(
                        np.multiply(input_a[i : i + b_h, j : j + b_w], input_b)
                    )
            return output

        def pad(
            input_a: NDArray[np.float32],
            input_b: NDArray[np.float32],
            step_size: Tuple[int, int],
            padding: Literal["valid", "full", "same"],
        ):
            pad_size = (0, 0)
            if padding == "same":

                def get_size(width, kernel_size, stride):
                    return np.ceil(
                        (stride * (width - 1) - width + kernel_size) / 2
                    ).astype(int)

                pad_size = (
                    get_size(input_a.shape[0], input_b.shape[0], step_size[0]),
                    get_size(input_a.shape[1], input_b.shape[1], step_size[1]),
                )
            elif padding == "full":
                pad_size = (input_b.shape[0] - 1, input_b.shape[1] - 1)
            elif padding == "valid":

                def get_max_valid_idx(size_a, size_b, stride):
                    res = (size_a - size_b) % stride
                    return -res if res != 0 else size_a

                input_a = input_a[
                    : get_max_valid_idx(
                        input_a.shape[0], input_b.shape[0], step_size[0]
                    ),
                    : get_max_valid_idx(
                        input_a.shape[1], input_b.shape[1], step_size[1]
                    ),
                    :,
                ]

            padded_input = np.zeros(
                (
                    input_a.shape[0] + 2 * pad_size[0],
                    input_a.shape[1] + 2 * pad_size[1],
                    input_a.shape[2],
                )
            )
            if pad_size != (0, 0):
                padded_input[
                    pad_size[0] : -pad_size[0], pad_size[1] : -pad_size[1], :
                ] = input_a
            else:
                padded_input = input_a

            return padded_input

        self.n_filters = n_filters

        def unpack(x):
            return (x, x) if isinstance(x, int) else x

        self.input_size = input_size
        self.kernel_size = unpack(kernel_size)
        self.step_size = unpack(step_size)

        self.padding = padding

        rnd = np.random.default_rng()
        self.weights = rnd.uniform(-1, 1, (*kernel_size, n_filters))
        self.d_weights = np.zeros(self.weights.shape)
        self.inputs = np.zeros(input_size)

        self.activation_function = activation_function

        def get_one_dim(w, k, s):
            return np.ceil((w - k) / s).astype(int) + 1

        def get_shape(shape_a, shape_b, step_size):
            return (
                get_one_dim(shape_a[0], shape_b[0], step_size[0]),
                get_one_dim(shape_a[1], shape_b[1], step_size[1]),
            )

        padded_shape = pad(
            self.inputs, self.weights, self.step_size, self.padding
        ).shape

        self.output_shape = (
            *get_shape(padded_shape, kernel_size, step_size),
            n_filters,
        )

        Layer.__init__(
            self,
            forward=forward,
            backward=backward,
            d_weights=np.zeros((self.weights.shape)),
            has_weights=True,
        )
