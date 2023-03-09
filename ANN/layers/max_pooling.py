"""Max Pooling Implementation"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from ANN.layers.layer import Layer


class MaxPooling2D(Layer):
    def __init__(self, pool_size: Tuple[int, int], step_size: Tuple[int, int]):
        self.pool_size = pool_size
        self.step_size = step_size
        self.x_idx = None
        self.y_idx = None

        def forward(inputs: NDArray[np.float32]) -> NDArray[np.float32]:
            """Computes forward pass on given inputs.

            Args:
                inputs (NDArray[np.float32]): Input array.
                Expect (n_samples, x_dim, y_dim, channels)

            Returns:
                NDArray[np.float32]: Outputs
            """
            self.x_idx = np.argmax(inputs, axis=1)
            self.y_idx = np.argmax(inputs[:, self.x_idx, :, :], axis=2)

            return inputs[:, self.x_idx, self.y_idx, :]

        def backward(error: NDArray[np.float32]) -> NDArray[np.float32]:
            """Computes backward pass on given error matrix.

            Args:
                error (NDArray[np.float32]): Error matrix

            Returns:
                NDArray[np.float32]: Input gradients.
            """
            output_shape = (
                error.shape[0],
                (error.shape[1] - 1) * self.step_size[0] + self.pool_size[0],
                (error.shape[2] - 1) * self.step_size[1] + self.pool_size[1],
                error.shape[3],
            )
