"""Neural Network layer object"""


from abc import abstractmethod
from typing import Tuple

import cupy as np
from numpy.typing import NDArray


class Layer:
    """Neural Network Layer Object"""

    def __init__(
        self,
        has_weights,
        weights,
        input_shape,
        output_shape,
        has_bias=False,
    ):
        self.has_weights = has_weights
        self.has_bias = has_bias

        self.weights = weights
        self.input_shape = input_shape
        self.output_shape = output_shape

    @abstractmethod
    def forward(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Forward pass on provided inputs

        Args:
            inputs (NDArray[np.float32]): Inputs

        Returns:
            NDArray[np.float32]: Layer activation
        """

    @abstractmethod
    def backward(self, gradient: NDArray[np.float32]) -> NDArray[np.float32]:
        """Backward pass using provided partial gradient

        Args:
            gradient (NDArray[np.float32]): Partial gradient to backpropagate

        Returns:
            NDArray[np.float32]: Partial gradient with regards to inputs.
        """

    @abstractmethod
    def initialize(self, input_shape: Tuple[int, ...]):
        """Initializa layer to match provided input shape

        Args:
            input_shape (Tuple[int,...]): Input shape to prepare the layer
        """
