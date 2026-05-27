"""
NovaX — GPU-Accelerated Math Library
------------------------------------
Lightweight, modular, and blazing-fast numerical computing library
with PyCUDA backend and automatic CPU fallback.

Author: Shohaib Shah
License: MIT
"""

__version__ = "0.2.0"
__author__ = "Shohaib Shah"

from .core import Tensor, GPU_AVAILABLE
from . import ops
from .dispatch import (
    add, sub, mul, div, pow, matmul,
    exp, log, sqrt, abs, neg,
    relu, sigmoid, tanh, softmax,
    sum, mean, max, min,
)
from .autograd import no_grad
from .cuda_graph import CUDAGraph
from .ops.launcher import launch_matmul_bias_relu


def set_default_device(device: str):
    """Set the default execution device: 'cpu' or 'gpu'."""
    from novax import dispatch
    dispatch.DEFAULT_DEVICE = device.lower()
    print(f"[NovaX] Default device set to: {dispatch.DEFAULT_DEVICE.upper()}")
