"""Helper functions to compute efficient correlation without using for loops by exploiting numpy's stride_tricks."""

from typing import Tuple

import cupy as cp
from cupy.typing import NDArray

from ANN.correlate.strided import get_strided_view


def corr2d_multi_in_out(
    input_a: NDArray[cp.float32],
    input_b: NDArray[cp.float32],
    step_size: Tuple[int, int],
) -> NDArray[cp.float32]:
    """Computes 2D correlation when input has multiple input channels and multiple output channels are desired.

    Args:
        input_a (NDArray [cp.float32]): Input to correlate.
        input_b (NDArray [cp.float32]): Input to correlate with.
        step_size (Tuple[int,int]): Step size for correlation.

    Returns:
        NDArray [cp.float32]: Correlation result.
    """
    return cp.einsum(
        "ijklmno,lmno->ijkl",
        get_strided_view(input_a=input_a, input_b=input_b, step_size=step_size),
        input_b,
    )
