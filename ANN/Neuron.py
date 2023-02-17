"""Implementation of a single neuron for a neural network.
"""

import numpy as np

class Neuron:
    """Implementation of a single neuron for a neural network.
    """
    def __init__(self, n_inputs, activation, loss):
        """Initialize the neuron with random weights and bias

        Args:
            n_inputs (int): size of input vector
            activation (Activation): Activation function object
            loss (Loss): Loss function object
        """
        rnd = np.random.default_rng()
        self.weights = rnd.uniform(-1,1,n_inputs)
        self.bias = rnd.uniform(-1,1,1)
        self.activation = activation
        self.loss = loss


    def set(self, weights, bias, activation, loss):
        """Set the properties of the neuron

        Args:
            weights (np.ndarray[np.float32]): Neuron's weight vector
            bias (np.float32): Neuron's bias
            activation (Activation): Activation function object
            loss (Loss): Loss function object
        """
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.loss = loss


    def forward(self, inputs):
        """Compute forward pass on neuron

        Args:
            inputs (np.ndarray[np.float32]): input vector

        Returns:
            np.float64: neuron activation
        """
        return self.activation.forward( np.dot(inputs, np.transpose(self.weights)) + self.bias)


    def backward(self, inputs, target, step_size):
        """Compute backward pass on neuron

        Args:
            inputs (np.ndarray[np.float32]): input vector
            output (np.float32): previous activation value of the neuron
            target (np.float32): target value
            step_size (np.float32): step size for gradient descent
        """
        output = self.forward(inputs)

        self.weights -= step_size * \
            (
                inputs *
                self.activation.backward((self.weights * inputs + self.bias)) *
                self.loss.backward(output, target)
            )

        self.bias -= step_size * \
            self.loss.backward(output, target)
