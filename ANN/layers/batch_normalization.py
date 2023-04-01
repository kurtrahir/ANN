"""Batch Normalization Implementation"""


from functools import reduce
from operator import mul
from typing import Optional, Tuple

import cupy as cp

from ANN.layers.initializers.gorlot import gorlot
from ANN.layers.layer import Layer


class BatchNormalization(Layer):
    """Batch normalization implementation"""

    def __init__(
        self,
        momentum: cp.float32 = 0.99,
        epsilon: cp.float32 = 1e-7,
        input_shape: Optional[Tuple[int, ...]] = None,
    ):
        self.weights = None
        self.d_weights = None
        self.bias = None
        self.d_bias = None
        self.output_shape = input_shape
        self.initialized = False
        if input_shape is not None:
            self.initialize(input_shape)

        self.epsilon = epsilon
        self.momentum = momentum

        Layer.__init__(
            self,
            has_weights=True,
            has_bias=True,
            input_shape=input_shape,
            output_shape=self.output_shape,
        )

    def forward(self, inputs, training):
        if not self.initialized:
            self.initialize(inputs.shape)
        self.inputs = inputs
        if training:
            self.mini_batch_mean = cp.mean(self.inputs, axis=0, keepdims=True)
            self.mini_batch_var = cp.var(self.inputs, axis=0, keepdims=True)
            self.mean_moving_average = (
                self.mean_moving_average * self.momentum
                + self.mini_batch_mean * (1 - self.momentum)
            )
            self.var_moving_average = (
                self.var_moving_average * self.momentum
                + self.mini_batch_var * (1 - self.momentum)
            )
        else:
            self.mini_batch_mean = self.mean_moving_average
            self.mini_batch_var = self.var_moving_average
        self.activation = (
            self.weights
            * (self.inputs - self.mini_batch_mean)
            / (self.mini_batch_var + self.epsilon)
            + self.bias
        )
        return self.activation

    def backward(self, gradient):
        # n_samples, n_features
        partial_gradient = gradient * self.weights

        d_var_1 = self.inputs - self.mini_batch_mean
        d_var_2 = -0.5 * cp.power(self.mini_batch_var + self.epsilon, -3 / 2)

        inverse_stdev = cp.divide(1, cp.sqrt(self.mini_batch_var + self.epsilon))

        d_var = cp.sum(partial_gradient * d_var_1 * d_var_2, axis=0, keepdims=True)

        d_mean = cp.sum(partial_gradient, axis=0, keepdims=True) * -inverse_stdev

        d_input_1 = cp.multiply(partial_gradient, inverse_stdev)
        d_input_2 = cp.multiply(
            d_var,
            cp.divide(2 * (self.inputs - self.mini_batch_mean), self.inputs.shape[0]),
        )
        d_input_3 = cp.divide(d_mean, self.inputs.shape[0])
        self.d_input = d_input_1 + d_input_2 + d_input_3

        self.d_bias = cp.sum(gradient, axis=0, keepdims=True)

        self.d_weights = cp.sum(
            cp.multiply(gradient, self.activation), axis=0, keepdims=True
        )
        return self.d_input

    def initialize(self, input_shape):
        self.input_shape = input_shape
        self.weights = cp.ones((1, *self.input_shape[1:]))
        self.d_weights = cp.zeros(self.weights.shape)
        self.bias = cp.zeros(self.weights.shape)
        self.d_bias = cp.zeros(self.bias.shape)
        self.output_shape = input_shape
        self.initialized = True

        self.mean_moving_average = cp.zeros(self.weights.shape)
        self.var_moving_average = cp.ones(self.weights.shape)
