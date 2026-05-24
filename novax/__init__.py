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
from .ops.launcher import launch_matmul_bias_relu, launch_matmul_bias, CUDAGraph


def set_default_device(device: str):
    """Set the default execution device: 'cpu' or 'gpu'."""
    from novax import dispatch
    dispatch.DEFAULT_DEVICE = device.lower()
    print(f"[NovaX] Default device set to: {dispatch.DEFAULT_DEVICE.upper()}")


def set_dtype(dtype: str):
    """
    Set the default tensor dtype for all subsequently created Tensors.

    Parameters
    ----------
    dtype : str
        ``'float32'`` (default) or ``'float16'`` / ``'fp16'`` / ``'half'``.
    """
    import numpy as np
    import novax.core as _core
    if dtype in ("float16", "fp16", "half"):
        _core._DEFAULT_DTYPE = np.float16
    else:
        _core._DEFAULT_DTYPE = np.float32
    print(f"[NovaX] Default dtype set to: {_core._DEFAULT_DTYPE}")
