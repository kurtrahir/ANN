"""Gorlot initializer
"""
from typing import Tuple

import cupy as np
from numpy.typing import NDArray


def gorlot(
    n_inputs: int, n_neurons: int, weight_shape: Tuple[int, ...]
) -> NDArray[np.float32]:
    """Generate weights according to gorlot

    Args:
        n_inputs (int): number of inputs
        n_neurons (int): number of neurons
        weight_shape (Tuple[int,...]): weight shape to generate

    Returns:
        NDArray[np.float32]: randomly generated weight vector
    """
    value = np.sqrt(6) / np.sqrt(n_inputs + n_neurons)
    return np.random.uniform(-value, value, weight_shape)
