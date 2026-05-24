import numpy as np

def sqrt(a):
    from novax.core import Tensor
    if np.any(a.data < 0):
        raise ValueError("sqrt() input must be non-negative")
    return Tensor(np.sqrt(a.data))
