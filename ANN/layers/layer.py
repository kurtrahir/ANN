"""Neural Network layer object"""


from abc import abstractmethod
from typing import Tuple

import cupy as cp
from cupy.typing import NDArray


class Layer:
    """Neural Network Layer Object"""

    def __init__(
        self,
        input_shape,
        output_shape,
        has_weights,
        has_bias,
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.has_weights = has_weights
        self.has_bias = has_bias

    @abstractmethod
    def forward(
        self, inputs: NDArray[cp.float32], training: bool
    ) -> NDArray[cp.float32]:
        """Forward pass on provided inputs

        Args:
            inputs (NDArray [cp.float32]): Inputs
            training (bool): Indicate whether this is part of netwrok training.

        Returns:
            NDArray [cp.float32]: Layer activation
        """

    @abstractmethod
    def backward(self, gradient: NDArray[cp.float32]) -> NDArray[cp.float32]:
        """Backward pass using provided partial gradient

        Args:
            gradient (NDArray [cp.float32]): Partial gradient to backpropagate

        Returns:
            NDArray [cp.float32]: Partial gradient with regards to inputs.
        """

    @abstractmethod
    def initialize(self, input_shape: Tuple[int, ...]):
        """Initializa layer to match provided input shape

        Args:
            input_shape (Tuple[int,...]): Input shape to prepare the layer
        """
