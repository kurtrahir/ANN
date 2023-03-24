from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray

from ANN.errors import ShapeError


def get_same_padding_shape(kernel_size):
    total_pad = kernel_size - 1
    half_pad = total_pad // 2
    if total_pad % 2 != 0:
        return half_pad, half_pad + 1
    return half_pad, half_pad


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

    if padding == "valid":
        return input_a[
            :,
            : get_max_valid_idx(input_a.shape[1], shape_b[0], step_size[0]),
            : get_max_valid_idx(input_a.shape[2], shape_b[1], step_size[1]),
            :,
        ]

    if padding == "full":
        pad_size = (shape_b[0] - 1, shape_b[1] - 1)
        return np.pad(
            input_a,
            [[0, 0], [pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [0, 0]],
            "constant",
        )

    if padding == "same":
        return np.pad(
            input_a,
            [
                [0, 0],
                [*get_same_padding_shape(shape_b[0])],
                [*get_same_padding_shape(shape_b[1])],
                [0, 0],
            ],
            "constant",
        )


def unpad(
    inputs: NDArray[np.float32],
    kernel_shape: Tuple[int, int],
    target_shape: Tuple[int, int, int, int],
    padding: Literal["full", "same", "valid"],
) -> NDArray[np.float32]:
    """Removes padding according to scheme used

    Args:
        inputs (NDArray[np.float32]): Array to unpad
        target_shape (Tuple[int,int,int,int]): Target shape of array
        padding (Literal["full", "same", "valid"]): Padding scheme used

    Returns:
        NDArray[np.float32]: Unpadded array
    """

    if len(inputs.shape) != 4:
        raise ShapeError(
            f"Expected input_a to have shape (n_samples, x_dim, y_dim, channels). \
            Received {inputs.shape=} instead."
        )

    if len(kernel_shape) != 2:
        raise ShapeError(
            f"Expected shape_b to be (x_dim, y_dim). \
            Received {kernel_shape=} instead."
        )

    if len(target_shape) != 4:
        raise ShapeError(
            f"Expected target_shape to have shape (n_samples, x_dim, y_dim, channels). \
            Received {target_shape=} instead."
        )

    if inputs.shape == target_shape:
        return inputs

    if padding == "full":
        pad_x = kernel_shape[0] - 1
        pad_y = kernel_shape[1] - 1
        return inputs[
            :, pad_x : inputs.shape[1] - pad_x, pad_y : inputs.shape[2] - pad_y, :
        ]

    if padding == "same":
        pad_x = get_same_padding_shape(kernel_shape[0])
        pad_y = get_same_padding_shape(kernel_shape[1])
        return inputs[
            :,
            pad_x[0] : min(inputs.shape[1] - pad_x[1], target_shape[1]),
            pad_y[0] : min(inputs.shape[2] - pad_y[1], target_shape[2]),
            :,
        ]

    if padding == "valid":
        pad_x = target_shape[1] - inputs.shape[1]
        pad_y = target_shape[2] - inputs.shape[2]
        return np.pad(inputs, [[0, 0], [0, pad_x], [0, pad_y], [0, 0]], "constant")
