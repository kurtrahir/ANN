"""Implementation of a single neuron for a neural network.
"""

import numpy as np
from numpy.typing import NDArray
from ANN.activation_functions import Activation
from ANN.loss_functions import Loss

class Neuron:
    """Implementation of a single neuron for a neural network.
    """
    def __init__(
            self,
            n_inputs : int,
            activation_function : Activation,
            loss_function : Loss
        ):
        """Initialize the neuron with random weights and bias

        Args:
            n_inputs (int): size of input vector
            activation (Activation): Activation function object
            loss (Loss): Loss function object
        """
        rnd = np.random.default_rng()
        self.weights = rnd.uniform(-1,1,n_inputs)
        self.bias = rnd.uniform(-1,1,n_inputs)
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.activation_value = 0
        self.linear_combination = np.zeros(self.weights.shape)
        self.gradients = np.zeros(self.weights.shape)

    def set(
            self,
            weights: NDArray[np.float32],
            bias: NDArray[np.float32],
            activation_function : Activation,
            loss_function : Loss
        ):
        """Set the properties of the neuron

        Args:
            weights (np.ndarray[np.float32]): Neuron's weight vector
            bias (np.float32): Neuron's bias
            activation (Activation): Activation function object
            loss (Loss): Loss function object
        """
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function
        self.loss_function = loss_function


    def forward(
            self,
            inputs: NDArray[np.float32]
        ):
        """Compute forward pass on neuron

        Args:
            inputs (np.ndarray[np.float32]): input vector

        Returns:
            np.float64: neuron activation
        """
        self.linear_combination = inputs * self.weights + self.bias
        self.activation_value = self.activation_function.forward(np.sum(self.linear_combination))
        return self.activation_value


    def backward(
            self,
            inputs : NDArray[np.float32],
            target: NDArray[np.float32],
            step_size : np.float32
        ):
        """Compute backward pass on neuron

        Args:
            inputs (np.ndarray[np.float32]): input vector
            output (np.float32): previous activation value of the neuron
            target (np.float32): target value
            step_size (np.float32): step size for gradient descent
        """
        _ = self.forward(inputs)

        update_term = self.activation_function.backward(self.linear_combination) * \
            self.loss_function.backward(self.activation_value, target) * \
            step_size

        self.gradients = inputs * update_term

        self.weights -= self.gradients

        self.bias -= update_term

        return self.gradients
