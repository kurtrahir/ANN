from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray

from ANN.errors import ShapeError


def pad(
    input_a: NDArray[np.float32],
    shape_b: Tuple[int, ...],
    step_size: Tuple[int, int],
    padding: Literal["valid", "full", "same"],
) -> NDArray[np.float32]:
    """Pad array to match desired convolution procedure when convolved with kernel of specified shape

    Args:
        input_a (NDArray[np.float32]): Array to be padded, expect (n_samples, x_dim, y_dim, channels)
        shape_b (Tuple[int,...]): Shape array will be convolved with expect (x_dim,y_dim)
        step_size (Tuple[int, int]): Step size to be used in convolution
        padding (Literal["valid", "full", "same"]): Padding strategy to employ

    Returns:
        NDArray[np.float32]: Padded array
    """

    if len(input_a.shape) != 4:
        raise ShapeError(
            f"Expected input_a to have shape (n_samples, x_dim, y_dim, channels). \
            Received {input_a.shape=} instead."
        )

    if len(shape_b) != 2:
        raise ShapeError(
            f"Expected shape_b to be (x_dim, y_dim). \
            Received {shape_b=} instead."
        )

    def get_max_valid_idx(size_a: int, size_b: int, stride: int) -> int:
        """Compute max idx to use in "valid" correlation.

        Args:
            size_a (int): Length of sequence to be correlated
            size_b (int): Length of sequence to be correlated with
            stride (int): Size of stride to use during correlation

        Returns:
            int: Max idx (<= size_a) ot use when a is convolved with b
            using provided stride.
        """
        res = (size_a - size_b) % stride
        return -res if res != 0 else size_a

    def get_same_padding_size(size_a, size_b, stride) -> int:
        """Compute size of padding to use in "same" correlation

        Args:
            size_a (int): Length of sequence to be correlated
            kernel_size (int): Length of sequence to be correlated with
            stride (int): Size of stride to use during correlation

        Returns:
            int: Size of padding to use for "same" correlation
        """
        return np.ceil((stride * (size_a - 1) - size_a + size_b) / 2).astype(int)

    pad_size = (0, 0)
    if padding == "full":
        pad_size = (shape_b[0] - 1, shape_b[1] - 1)
    elif padding == "same":
        pad_size = (
            get_same_padding_size(input_a.shape[1], shape_b[0], step_size[0]),
            get_same_padding_size(input_a.shape[2], shape_b[1], step_size[1]),
        )
    elif padding == "valid":
        return input_a[
            :,
            : get_max_valid_idx(input_a.shape[1], shape_b[0], step_size[0]),
            : get_max_valid_idx(input_a.shape[2], shape_b[1], step_size[1]),
            :,
        ]
    padded_input = np.pad(
        input_a,
        [[0, 0], [pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [0, 0]],
        "constant",
    )
    return padded_input
