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
        self.input_strides = None
        self.output_shape = None
        self.strided_shape = None
        self.strided_strides = None
        self.argmax_shape = None
        self.idx = None
        self.initialized = False

        def forward(inputs: NDArray[np.float32]) -> NDArray[np.float32]:
            """Compute forward pass on provided inputs.

            Args:
                inputs (NDArray[np.float32]): Input array to reduce, expect (n_sampled, x_dim, y_dim, n_channels)

            Returns:
                NDArray[np.float32]: Max-Pooled array (n_sampled, x_dim, y_dim, n_channels)
            """
            if not self.initialized:
                self.initialize_weights(inputs.shape)
            # Store input shape and strides for backward pass
            self.input_shape = inputs.shape
            self.input_strides = inputs.strides
            # Get strided view to calculate local maxima
            strided_inputs = get_strided_view(
                inputs,
                np.zeros((1, *self.kernel_size, inputs.shape[-1])),
                self.step_size,
            )
            # Store strided shape and strides
            self.strided_shape = strided_inputs.shape
            self.strided_strides = strided_inputs.strides
            # Set shape for obtaining argmax - flatten 2 pool dims as argmax uses 1 dimension
            # Store shape for later setting
            self.argmax_shape = (
                strided_inputs.shape[:4] + (-1,) + (strided_inputs.shape[-1],)
            )
            strided_data_reshaped = strided_inputs.reshape(self.argmax_shape)
            # Get idx of local max elements
            self.idx = np.argmax(strided_data_reshaped, axis=4, keepdims=True)
            # Return pooled array
            return np.max(strided_inputs, axis=(4, 5))

        def backward(error: NDArray[np.float32]) -> NDArray[np.float32]:
            """Compute backward pass on provided error array.

            Args:
                error (NDArray[np.float32]): Error array (expect (n_sampled, x_dim, y_dim, n_channels))

            Returns:
                NDArray[np.float32]: Input gradients (n_sampled, x_dim, y_dim, n_channels)
            """
            # Create an array with same dimensions as input,
            # and create a view with the same shape and strides as when the idx was obtained
            d_inputs = np.lib.stride_tricks.as_strided(
                np.zeros(self.input_shape), self.argmax_shape, self.strided_strides
            )
            # Set elements that were propagated in otherwise 0 array
            d_inputs[self.idx] = error
            # Reset view to normal array.
            return np.lib.stride_tricks.as_strided(
                d_inputs, self.input_shape, self.input_strides
            )

        def initialize_weights(input_shape: Tuple[int, int, int, int]) -> None:
            """Initialize layer

            Args:
                input_shape (Tuple[int,int,int,int]): Input shape (expect n_sampled, x_dim, y_dim, n_channels)
            """
            self.input_shape = input_shape
            self.output_shape = (input_shape[0],) + get_shape(
                input_shape[1:], (1, *self.kernel_size, input_shape[-1]), self.step_size
            )
            self.initialized = True

        Layer.__init__(
            self,
            forward=forward,
            backward=backward,
            initialize_weights=initialize_weights,
            has_bias=False,
            has_weights=False,
            input_shape=None,
            output_shape=None,
            weights=None,
        )
