"""Convolutional layer implementation
"""

from typing import Literal, Tuple, Union

import numpy as np

from ANN.activation_functions import Activation
from ANN.layers import Layer


class Conv2D(Layer):
    def __init__(
        self,
        n_filters: int,
        kernel_size: Union[int, tuple[int, int]],
        activation_function: Activation,
        step_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int], Literal["Same"]],
    ):
        self.n_filters = n_filters

        def unpack(x):
            return (x, x) if isinstance(x, int) else x

        self.kernel_size = unpack(kernel_size)
        self.step_size = unpack(step_size)
        self.padding = unpack(padding)
        rnd = np.random.default_rng()
        self.kernels = rnd.uniform(-1, 1, (n_filters, *kernel_size))
        self.activation_function = activation_function

        def forward(inputs):
            if self.padding == "Same":

                def get_size(w, k, s):
                    return np.ceil((s * (w - 1) - w + k) / 2).astype(int)

                self.padding = (
                    get_size(inputs.shape[1], kernel_size[0], step_size[0]),
                    get_size(inputs.shape[2], kernel_size[1], step_size[1]),
                )
            padded_inputs = np.zeros(
                (
                    inputs.shape[0],
                    inputs.shape[1] + 2 * self.padding[0],
                    inputs.shape[2] + 2 * self.padding[1],
                )
            )
            if padding != (0, 0):
                padded_inputs[
                    :,
                    self.padding[0] : -self.padding[0],
                    self.padding[1] : -self.padding[1],
                ] = inputs
            else:
                padded_inputs = inputs
            x_shape = np.ceil(
                (inputs.shape[1] - self.kernel_size[0] + 2 * self.padding[0])
                / step_size[0]
            ).astype(int)
            y_shape = np.ceil(
                (inputs.shape[2] - self.kernel_size[1] + 2 * self.padding[1])
                / step_size[1]
            ).astype(int)
            outputs = np.zeros((n_filters * inputs.shape[0], x_shape, y_shape))
            for i in range(0, x_shape):
                for j in range(0, y_shape):
                    for k in range(inputs.shape[0]):
                        x_idx = i * step_size[0]
                        y_idx = j * step_size[1]
                        z_idx = k * n_filters
                        outputs[z_idx : z_idx + n_filters, i, j] = np.sum(
                            np.multiply(
                                padded_inputs[
                                    k,
                                    x_idx : x_idx + kernel_size[0],
                                    y_idx : y_idx + kernel_size[1],
                                ],
                                self.kernels,
                            ),
                            axis=(1, 2),
                        )
            return self.activation_function.forward(outputs)

        def backward():
            return 0

        Layer.__init__(
            self,
            forward=forward,
            backward=backward,
            d_weights=np.zeros((self.kernels.shape)),
        )
