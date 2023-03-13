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
            # Store input shape and strides for backward pass
            self.input_shape = inputs.shape
            self.input_strides = inputs.strides
            # Get strided view to calculate local maxima, remove dimension used for multiple
            # feature map correlation
            strided_inputs = get_strided_view(
                inputs,
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
            # Obtain index
            self.idx = np.argmax(strided_data_reshaped, axis=3)
            # Return pooled array
            return np.max(strided_inputs, axis=(3, 4))

        def backward(error: NDArray[np.float32]) -> NDArray[np.float32]:
            """Compute backward pass on provided error array.

            Args:
                error (NDArray[np.float32]): Error array (expect (n_sampled, x_dim, y_dim, n_channels))

            Returns:
                NDArray[np.float32]: Input gradients (n_sampled, x_dim, y_dim, n_channels)
            """
            # Create an array with same dimensions as input
            d_inputs = np.zeros(self.input_shape)
            # Iterate over array and set gradients using index stored in self.idx
            for s in range(error.shape[0]):
                for x in range(error.shape[1]):
                    xs = x * self.step_size[0]
                    xe = xs + self.kernel_size[0]
                    for y in range(error.shape[2]):
                        ys = y * self.step_size[1]
                        ye = ys + self.kernel_size[1]
                        for z in range(error.shape[-1]):
                            d_inputs[s, xs:xe, ys:ye, z][
                                self.idx[s, x, y, z] // 2, self.idx[s, x, y, z] % 2
                            ] = error[s, x, y, z]
            # Return gradients
            return d_inputs

        def initialize_weights(input_shape: Tuple[int, int, int]) -> None:
            """Initialize layer

            Args:
                input_shape (Tuple[int,int,int]): Input shape (expect x_dim, y_dim, n_channels)
            """
            self.input_shape = input_shape
            self.output_shape = get_shape(
                input_shape, (1, *self.kernel_size, input_shape[-1]), self.step_size
            ) + (input_shape[-1],)
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
