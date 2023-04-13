"""Generic Callback Class"""

from abc import ABC, abstractmethod


class Callback(ABC):
    @abstractmethod
    def __init__(self):
        """Initialize callback object"""

    @abstractmethod
    def call(self, model):
        """Execute callback operation

        Args:
            model (Model): The model being trained
        """
