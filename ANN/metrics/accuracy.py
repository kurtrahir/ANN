"""Accuracy metric"""

from functools import reduce
from operator import mul

import cupy as cp
from cupy.typing import NDArray


def accuracy(pred: NDArray[cp.float32], true: NDArray[cp.float32]):
    idx = true.astype(bool)
    return cp.sum((pred[idx] == true[idx]).astype(int)) / cp.sum(pred)
