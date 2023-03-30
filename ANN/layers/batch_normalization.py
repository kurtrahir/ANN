"""Batch Normalization Implementation"""


import cupy as cp

from ANN.layers.layer import Layer


class BatchNormalization(Layer):
    """Batch normalization implementation"""

    def __init__(self, gamma: cp.float32, beta: cp.float32, epsilon: cp.float32 = 1e-7):
        self.input_shape = None
        self.weights = cp.array([gamma, beta])
        self.d_weights = cp.zeros([1, 2])
        Layer.__init__(
            self,
            has_weights=True,
            weights=None,
            input_shape=None,
            output_shape=None,
        )

    def forward(self, inputs):
        self.inputs = inputs
        self.mini_batch_mean = cp.mean(self.inputs, axis=0)
        self.mini_batch_var = cp.var(self.inputs, axis=0)
        self.activation = (
            self.weights[0]
            * (self.inputs - self.mini_batch_mean)
            / (self.mini_batch_var + self.epsilon)
            + self.weights[1]
        )
        return self.activation

    def backward(self, gradient):
        partial_gradient = gradient * self.weights[0]

        self.d_var = cp.sum(
            gradient
            * cp.dot(
                cp.dot(gradient, (self.inputs - self.mini_batch_mean)),
                -0.5 * cp.power((self.mini_batch_var + self.epsilon), -1.5),
            ),
            axis=0,
        )
        self.d_mean = cp.sum(
            cp.dot(
                partial_gradient,
                cp.divide(-1, cp.sqrt(self.mini_batch_var + self.epsilon)),
            )
        )
        self.d_input = cp.dot(
            partial_gradient, cp.divide(1, cp.sqrt(self.mini_batch_var + self.epsilon))
        ) + cp.dot(
            self.d_var,
            cp.divide(2 * (self.inputs - self.mini_batch_mean), self.inputs.shape[0])
            + cp.dot(self.d_mean, 1 / self.inputs.shape[0]),
        )
        self.d_weights[0] = cp.sum(cp.dot(partial_gradient, self.activation), axis=0)
        self.d_weights[1] = cp.sum(partial_gradient, axis=0)
        return self.d_input

    def initialize(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
