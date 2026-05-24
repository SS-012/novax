import numpy as np

"""
CPU fallback elementwise subtraction operation
"""

def sub(a, b):
    from novax.core import Tensor
    return Tensor(a.data - b.data)
