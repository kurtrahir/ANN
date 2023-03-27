"""Utility function for calculating the output shape of strided correlation."""

from math import floor
from typing import Tuple

from ANN.errors.shapeError import ShapeError


def get_shape(
    shape_a: Tuple[int, ...], shape_b: Tuple[int, ...], step_size: Tuple[int, int]
) -> Tuple[int, ...]:
    """Return output shape resulting from correlating a with b

    Args:
        shape_a (Tuple[int, ...]): Shape of array to be correlated (x_dim, y_dim, ...)
        shape_b (Tuple[int,...]): Shape of array to be correlated with (x_dim, y_dim,...) or (n_samples, x_dim, y_dim,...)
        step_size (Tuple[int, int]): Step size in x_dim, y_dim

    Returns:
        Tuple[int,int]: Shape of output array (x_dim, y_dim)
    """

    def get_square(shape: Tuple[int, ...]):
        if len(shape) == 4:
            return shape[1:3]
        if len(shape) == 3:
            return shape[:2]
        if len(shape) == 2:
            return shape
        else:
            raise ShapeError(f"Could not identify square in provided shape: {shape=}")

    square_a = get_square(shape_a)
    square_b = get_square(shape_b)

    def validate_stride(size_a, size_b, stride):
        if (size_a - size_b) % stride != 0:
            raise ShapeError("Invalid stride setting.")

    validate_stride(square_a[0], square_b[0], step_size[0])
    validate_stride(square_a[1], square_b[1], step_size[1])
    corr_square = (
        floor((square_a[0] - square_b[0]) / step_size[0] + 1),
        floor((square_a[1] - square_b[1]) / step_size[1] + 1),
    )
    # If multi-sample, multi-channel
    if len(shape_a) == 4 and len(shape_b) == 4:
        if shape_a[3] == shape_b[3]:
            return (shape_a[0],) + corr_square + (shape_b[0],)
        else:
            raise ShapeError(
                f"Expected number of channels to match. Got {shape_a=} and {shape_b=}."
            )
    # Case single sample, multichannel
    if len(shape_b) == 3 and len(shape_a) == 3:
        if shape_b[2] == shape_a[2]:
            return corr_square
        else:
            raise ShapeError(
                f"Expected number of channels to match. Got {shape_a=} and {shape_b=}"
            )
    # Case single sample, single channel
    if len(shape_b) == 2 and len(shape_a) == 2:
        return corr_square
    # Case multisample, single channel.
    if len(shape_b) == 2 and len(shape_a) == 3:
        return (shape_a[0],) + corr_square
