"""Helper functions to compute efficient correlation without using for loops by exploiting numpy's stride_tricks."""

from typing import Tuple

import cupy as np
from cupy.typing import NDArray

from ANN.correlate.strided import get_strided_view


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
    return np.einsum(
        "ijklmno,lmno->ijkl",
        get_strided_view(input_a=input_a, input_b=input_b, step_size=step_size),
        input_b,
    )
