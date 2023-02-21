"""Stochastic Gradient Descent optimizer"""
from ANN.layers import Layer
from ANN.optimizers import Optimizer
import numpy as np
from numpy.typing import NDArray


class SGD(Optimizer):
    """Stochastic Gradient Descent Optimizer
    """
    def __init__(self, step_size):

        def get_update(
                layer : Layer,
                targets : NDArray[np.float32]
        ):

            linear_combination = np.multiply(layer.inputs.T, layer.weights)
            output = layer.activation_function.forward(
                np.sum(linear_combination, axis=0))

            update_term = layer.activation_function.backward(linear_combination) * \
                layer.loss_function.backward(output, targets) * \
                step_size

            return np.multiply(layer.inputs.T, update_term)


        Optimizer.__init__(
            self,
            get_update=get_update
        )