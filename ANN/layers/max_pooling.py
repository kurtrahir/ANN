"""Max Pooling Implementation"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from ANN.correlate.shape import get_shape
from ANN.correlate.strided import get_strided_view
from ANN.layers.layer import Layer


class MaxPool2D(Layer):
    def __init__(self, kernel_size: Tuple[int, int], step_size: Tuple[int, int]):
        self.kernel_size = kernel_size
        self.step_size = step_size
        self.input_shape = None
        self.inputs = None
        self.padded_shape = None
        self.output_shape = None
        self.idx = None
        self.initialized = False
        self.pad = None

        self.strided_shape = None
        self.strided_strides = None

        Layer.__init__(
            self,
            has_bias=False,
            has_weights=False,
            input_shape=None,
            output_shape=None,
            weights=None,
        )

    def forward(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute forward pass on provided inputs.

        Args:
            inputs (NDArray[np.float32]): Input array to reduce, expect (n_sampled, x_dim, y_dim, n_channels)

        Returns:
            NDArray[np.float32]: Max-Pooled array (n_sampled, x_dim, y_dim, n_channels)
        """
        if not self.initialized or self.input_shape != inputs.shape:
            self.initialize(inputs.shape)
        self.inputs[:, : inputs.shape[1], : inputs.shape[2], :] = inputs
        # Get strided view to calculate local maxima, remove dimension used for multiple
        # feature map correlation
        strided_inputs = get_strided_view(
            self.inputs,
            np.zeros((1, *self.kernel_size, inputs.shape[-1])),
            self.step_size,
        )[..., 0, :, :, :]
        strided_data_reshaped = strided_inputs.reshape(
            strided_inputs.shape[0],
            strided_inputs.shape[1],
            strided_inputs.shape[2],
            -1,
            strided_inputs.shape[-1],
        )
        self.strided_shape = strided_inputs.shape
        self.strided_strides = strided_inputs.strides
        # Obtain index
        self.idx = np.argmax(strided_data_reshaped, axis=3)
        # Return pooled array
        return np.max(strided_inputs, axis=(3, 4))

    def backward(self, gradient: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute backward pass on provided error array.

        Args:
            gradient (NDArray[np.float32]): Error array (expect (n_sampled, x_dim, y_dim, n_channels))

        Returns:
            NDArray[np.float32]: Input gradients (n_sampled, x_dim, y_dim, n_channels)
        """
        # Create an array with same dimensions as input
        d_inputs = np.zeros(
            (
                gradient.shape[0],
                self.input_shape[1] + self.pad[0],
                self.input_shape[2] + self.pad[1],
                gradient.shape[3],
            )
        )
        # Iterate over array and set gradients using index stored in self.idx
        d_inputs = np.lib.stride_tricks.as_strided(
            d_inputs, self.strided_shape, self.strided_strides
        )
        s_idx = np.arange(d_inputs.shape[0])[:, None, None, None]
        x_idx = np.arange(d_inputs.shape[1])[:, None, None]
        y_idx = np.arange(d_inputs.shape[2])[:, None]
        c_idx = np.arange(d_inputs.shape[5])
        d_inputs[
            s_idx,
            x_idx,
            y_idx,
            self.idx // self.kernel_size[0],
            self.idx % self.kernel_size[1],
            c_idx,
        ] = gradient
        d_inputs = np.lib.stride_tricks.as_strided(
            d_inputs, self.inputs.shape, self.inputs.strides
        )
        # Return gradients
        return d_inputs[:, : self.input_shape[1], : self.input_shape[2], :]

    def initialize(self, input_shape: Tuple[int, int, int, int]) -> None:
        """Initialize layer

        Args:
            input_shape (Tuple[int,int,int]): Input shape, expect (n_samples, x_dim, y_dim, n_channels)
        """
        self.input_shape = input_shape

        def calc_pad(x, k, s):
            res = (x - k) % s
            if res != 0:
                return k - res
            return 0

        self.pad = (
            calc_pad(input_shape[1], self.kernel_size[0], self.step_size[0]),
            calc_pad(input_shape[2], self.kernel_size[1], self.step_size[1]),
        )
        self.padded_shape = (
            input_shape[0],
            input_shape[1] + self.pad[0],
            input_shape[2] + self.pad[1],
            input_shape[3],
        )
        self.inputs = np.zeros(self.padded_shape)
        self.output_shape = get_shape(
            self.padded_shape,
            (self.padded_shape[-1],) + self.kernel_size + (self.padded_shape[-1],),
            self.step_size,
        )
        self.initialized = True
