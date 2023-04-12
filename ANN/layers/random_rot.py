"""Randomly rotate an input image by multiples of 90
"""

from typing import Tuple

import cupy as cp
import numpy as np
from cupy.typing import NDArray
from cupyx.scipy.ndimage import rotate

from ANN.layers.layer import Layer


class RandomRot(Layer):
    def __init__(self, max_angle):
        self.max_angle = max_angle
        Layer.__init__(
            self, input_shape=None, output_shape=None, has_bias=False, has_weights=False
        )
        self.rnd = np.random.default_rng()

    def forward(self, inputs: NDArray[cp.float32], training: bool = False):
        if training is False:
            return inputs
        angles = self.rnd.integers(-self.max_angle, self.max_angle, inputs.shape[0])
        for i in range(len(angles)):
            temp = rotate(input=inputs[i], angle=angles[i], order=0)
            if temp.shape == inputs.shape[1:]:
                inputs[i] = temp
                continue
            x_offset = (temp.shape[0] - inputs.shape[1]) // 2
            x_rem = (temp.shape[0] - inputs.shape[1]) % 2
            y_offset = (temp.shape[1] - inputs.shape[2]) // 2
            y_rem = (temp.shape[1] - inputs.shape[2]) % 2
            inputs[i] = temp[
                x_offset : -(x_offset + x_rem), y_offset : -(y_offset + y_rem), :
            ]
        return inputs

    def backward(self, gradients: NDArray[cp.float32]):
        return gradients

    def initialize(self, input_shape: Tuple[int, ...]):
        self.input_shape = input_shape
        self.output_shape = input_shape
