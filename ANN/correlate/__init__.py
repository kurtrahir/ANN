"""Helper functions to compute efficient correlation without using for loops by exploiting numpy's stride_tricks."""
from .correlate import corr2d, corr2d_multi_in, corr2d_multi_in_out
