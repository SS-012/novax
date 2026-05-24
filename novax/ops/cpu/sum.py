import numpy as np

def sum(a):
    from novax.core import Tensor
    return Tensor(np.array([np.sum(a.data)], dtype=np.float32))
