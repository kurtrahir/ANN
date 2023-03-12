"""Utility function for calculating the output shape of strided correlation."""

from typing import Tuple

import numpy as np


def get_shape(
    shape_a: Tuple[int, int], shape_b: Tuple[int, ...], step_size: Tuple[int, int]
) -> Tuple[int, int]:
    """Return output shape resulting from correlating a with b


    Args:
        shape_a (Tuple[int, int]): Shape of array to be correlated (x_dim, y_dim, ...)
        shape_b (Tuple[int,...]): Shape of array to be correlated with (x_dim, y_dim,...) or (n_samples, x_dim, y_dim,...)
        step_size (Tuple[int, int]): Step size in x_dim, y_dim

    Returns:
        Tuple[int,int]: Shape of output array (x_dim, y_dim)
    """
    if len(shape_b) < 4:
        return (
            np.ceil((shape_a[0] - shape_b[0]) / step_size[0]).astype(int) + 1,
            np.ceil((shape_a[1] - shape_b[1]) / step_size[1]).astype(int) + 1,
        )
    return (
        np.ceil((shape_a[0] - shape_b[1]) / step_size[0]).astype(int) + 1,
        np.ceil((shape_a[1] - shape_b[2]) / step_size[1]).astype(int) + 1,
    )
