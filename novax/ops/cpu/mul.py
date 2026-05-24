import numpy as np

"""
CPU fallback elementwise multiplication operation
"""

def mul(a, b):
    from novax.core import Tensor
    return Tensor(a.data * b.data)
