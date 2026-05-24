import numpy as np

def max(a):
    from novax.core import Tensor
    return Tensor(np.array([np.max(a.data)], dtype=np.float32))
