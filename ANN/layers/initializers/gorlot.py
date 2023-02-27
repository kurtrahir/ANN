"""Gorlot initializer
"""
import numpy as np
from numpy.typing import NDArray


def gorlot(n_inputs: int, n_neurons: int, n_weights: int) -> NDArray[np.float32]:
    """Generate weights according to gorlot

    Args:
        n_inputs (int): number of inputs
        n_neurons (int): number of neurons
        n_weights (int): number of weights to generate

    Returns:
        NDArray[np.float32]: randomly generated weight vector
    """
    rnd = np.random.default_rng()
    value = np.sqrt(6) / np.sqrt(n_inputs + n_neurons)
    return rnd.uniform(-value, value, n_weights)
