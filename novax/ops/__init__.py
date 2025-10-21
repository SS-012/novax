from novax.core import GPU_AVAILABLE

if GPU_AVAILABLE:
    from .gpu import *
else:
    from .cpu import *

__all__ = ["add", "sub", "mul", "div"]