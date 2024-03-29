"""Utility function for obtaining a strided view of an array for efficient kernel shaped access (e.g. correlation or maxpool 2d)
    """

from typing import Tuple

import cupy as cp
from cupy.typing import NDArray

from ANN.correlate.shape import get_shape


def get_strided_view(
    input_a: NDArray[cp.float32], input_b: NDArray[cp.float32], step_size
) -> NDArray[cp.float32]:
    """Returns a strided view of a numpy array designed for efficient correlation.

    Args:
        input_a (NDArray [cp.float32]): Input to be correlated.
        input_b (NDArray [cp.float32]): Input to be correlated with.
        step_size (_type_): Step size for correlation.

    Returns:
        NDArray [cp.float32]: Strided view of input a.
    """
    return cp.lib.stride_tricks.as_strided(
        x=cp.asarray(input_a),
        shape=get_shape(input_a.shape, input_b.shape, step_size=step_size)
        + input_b.shape[1:],
        strides=get_strides(input_a=input_a, input_b=input_b, step_size=step_size),
    )


def get_strides(
    input_a: NDArray[cp.float32], input_b: NDArray[cp.float32], step_size
) -> Tuple[int, ...]:
    """Computes the stride size for the creation of a strided view of input_a for efficient correlation with input b.

    Args:
        input_a (NDArray [cp.float32]): Input to be correlated.
        input_b (NDArray [cp.float32]): Input to be correlated with.
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
