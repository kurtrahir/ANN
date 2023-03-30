"""Gorlot initializer
"""
from typing import Tuple

import cupy as cp
from cupy.typing import NDArray


def gorlot(
    n_inputs: int, n_neurons: int, weight_shape: Tuple[int, ...]
) -> NDArray[cp.float32]:
    """Generate weights according to gorlot

    Args:
        n_inputs (int): number of inputs
        n_neurons (int): number of neurons
        weight_shape (Tuple[int,...]): weight shape to generate

    Returns:
        NDArray [cp.float32]: randomly generated weight vector
    """
    value = cp.sqrt(6) / cp.sqrt(n_inputs + n_neurons)
    return cp.random.uniform(-value, value, weight_shape, dtype=cp.float32)
