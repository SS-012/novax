import numpy as np

def log(a):
    from novax.core import Tensor
    if np.any(a.data <= 0):
        raise ValueError("log() input must be strictly positive")
    return Tensor(np.log(a.data))
