import numpy as np

"""
CPU fallback elementwise division operation
"""

def div(a, b):
    from novax.core import Tensor
    if np.any(b.data == 0):
        raise ZeroDivisionError("Division by zero in tensor operation")
    return Tensor(a.data / b.data)
