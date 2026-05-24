import numpy as np

"""
CPU fallback elementwise addition operation
"""

def add(a, b):
    from novax.core import Tensor
    return Tensor(a.data + b.data)
