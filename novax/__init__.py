"""
NovaX — GPU-Accelerated Math Library
------------------------------------
Lightweight, modular, and blazing-fast numerical computing library
with PyCUDA backend and automatic CPU fallback.

Author: Shohaib Shah
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Shohaib Shah"

from .core import Tensor, GPU_AVAILABLE
from . import ops
from .dispatch import add, sub, mul, div

def set_default_device(device: str):
    """
    Set the default device for operations: "cpu" or "gpu"
    """
    from novax import dispatch
    dispatch.DEFAULT_DEVICE = device.lower()
    print(f"[NovaX] Default device set to: {dispatch.DEFAULT_DEVICE.upper()}")
