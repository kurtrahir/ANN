"""Helper functions to compute efficient correlation without using for loops by exploiting numpy's stride_tricks."""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def get_shape(shape_a, shape_b, step_size):
    if len(shape_b) < 4:
        return (
            np.ceil((shape_a[0] - shape_b[0]) / step_size[0]).astype(int) + 1,
            np.ceil((shape_a[1] - shape_b[1]) / step_size[1]).astype(int) + 1,
        )
    return (
        np.ceil((shape_a[0] - shape_b[1]) / step_size[0]).astype(int) + 1,
        np.ceil((shape_a[1] - shape_b[2]) / step_size[1]).astype(int) + 1,
    )


def corr2d_multi_in_out(
    input_a: NDArray[np.float32],
    input_b: NDArray[np.float32],
    step_size: Tuple[int, int],
) -> NDArray[np.float32]:
    """Computes 2D correlation when input has multiple input channels and multiple output channels are desired.

    Args:
        input_a (NDArray[np.float32]): Input to correlate.
        input_b (NDArray[np.float32]): Input to correlate with.
        step_size (Tuple[int,int]): Step size for correlation.

    Returns:
        NDArray[np.float32]: Correlation result.
    """
    return np.einsum("ijklmno,lmno->ijkl", get_strided_view(**locals()), input_b)


def get_strided_view(
    input_a: NDArray[np.float32], input_b: NDArray[np.float32], step_size
) -> NDArray[np.float32]:
    """Returns a strided view of a numpy array designed for efficient correlation.

    Args:
        input_a (NDArray[np.float32]): Input to be correlated.
        input_b (NDArray[np.float32]): Input to be correlated with.
        step_size (_type_): Step size for correlation.

    Returns:
        NDArray[np.float32]: Strided view of input a.
    """
    return np.lib.stride_tricks.as_strided(
        x=input_a,
        shape=(
            (input_a.shape[0],)
            + get_shape(input_a.shape[1:], input_b.shape, step_size=step_size)
            + input_b.shape
        ),
        strides=get_strides(**locals()),
        writeable=False,
    )


def get_strides(
    input_a: NDArray[np.float32], input_b: NDArray[np.float32], step_size
) -> Tuple[int, ...]:
    """Computes the stride size for the creation of a strided view of input_a for efficient correlation with input b.

    Args:
        input_a (NDArray[np.float32]): Input to be correlated.
        input_b (NDArray[np.float32]): Input to be correlated with.
        step_size (_type_): Step size for correlation.

    Returns:
        Tuple[int,...]: Stride sizes.
    """
    if len(input_b.shape) < 4:
        return (
            input_a.strides[0],
            input_a.strides[1] * step_size[0],
            input_a.strides[2] * step_size[1],
            *input_a.strides[1:],
        )
    return (
        input_a.strides[0],
        input_a.strides[1] * step_size[0],
        input_a.strides[2] * step_size[1],
        0,
        *input_a.strides[1:],
    )
