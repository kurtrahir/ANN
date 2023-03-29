from typing import Tuple

import cupy as np
from cupy.typing import NDArray


def dilate(
    array: NDArray[np.float32], step_size: Tuple[int, int]
) -> NDArray[np.float32]:
    """Dilate array using provided step size

    Args:
        array (NDArray[np.float32]): Array to dilate, expects (n_samples, x_dim, y_dim,...)
        step_size (Tuple[int, int]): Step size used in forward convolution
        to decide the appropriate dilation size.

    Returns:
        NDArray[np.float32]: Dilated Array
    """
    output = np.zeros(
        (
            array.shape[0],
            array.shape[1] + (step_size[0] - 1) * (array.shape[1] - 1),
            array.shape[2] + (step_size[1] - 1) * (array.shape[2] - 1),
            *array.shape[3:],
        )
    )

    output[:, :: step_size[0], :: step_size[1]] = array

    return output
