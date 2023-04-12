"""Randomly rotate an input image by multiples of 90
"""

from typing import Tuple

import cupy as cp
import numpy as np
from cupy.typing import NDArray
from cupyx.scipy.ndimage import shift

from ANN.layers.layer import Layer


class RandomShift(Layer):
    def __init__(self, shift_percentage):
        self.shift_percentage = shift_percentage
        Layer.__init__(
            self, input_shape=None, output_shape=None, has_bias=False, has_weights=False
        )
        self.rnd = np.random.default_rng()

    def forward(self, inputs: NDArray[cp.float32], training: bool = False):
        if training is False:
            return inputs
        x_shift = self.rnd.random(inputs.shape[0]) * (
            self.shift_percentage * inputs.shape[1]
        )
        y_shift = self.rnd.random(inputs.shape[0]) * (
            self.shift_percentage * inputs.shape[2]
        )

        for i in range(inputs.shape[0]):
            inputs[i] = shift(inputs[i], shift=[x_shift[i], y_shift[i], 0], order=0)

        return inputs

    def backward(self, gradients: NDArray[cp.float32]):
        return gradients

    def initialize(self, input_shape: Tuple[int, ...]):
        self.input_shape = input_shape
        self.output_shape = input_shape
