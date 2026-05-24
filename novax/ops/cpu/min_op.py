import numpy as np

def min(a):
    from novax.core import Tensor
    return Tensor(np.array([np.min(a.data)], dtype=np.float32))
