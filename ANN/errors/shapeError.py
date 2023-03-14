"""Error to throw when the shape is incorrect.
"""


class ShapeError(ValueError):
    """Basic Shape Error"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
