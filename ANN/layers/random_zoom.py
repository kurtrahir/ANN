"""Randomly rotate an input image by multiples of 90
"""

from typing import Tuple

import cupy as cp
import numpy as np
from cupy.typing import NDArray
from cupyx.scipy.ndimage import zoom

from ANN.layers.layer import Layer


class RandomZoom(Layer):
    def __init__(self, zoom_factor):
        self.zoom_factor = zoom_factor
        Layer.__init__(
            self, input_shape=None, output_shape=None, has_bias=False, has_weights=False
        )
        self.rnd = np.random.default_rng()

    def forward(self, inputs: NDArray[cp.float32], training: bool = False):
        if training is False:
            return inputs
        zooms = self.rnd.random(inputs.shape[0]) * self.zoom_factor + 1

        for i in range(inputs.shape[0]):
            inputs[i] = zoom(inputs[i], zoom=[zooms[i], zooms[i], 1], order=0)[
                : inputs.shape[1], : inputs.shape[2], :
            ]

        return inputs

    def backward(self, gradients: NDArray[cp.float32]):
        return gradients

    def initialize(self, input_shape: Tuple[int, ...]):
        self.input_shape = input_shape
        self.output_shape = input_shape
