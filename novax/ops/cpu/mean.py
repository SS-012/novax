import numpy as np

def mean(a):
    from novax.core import Tensor
    return Tensor(np.array([np.mean(a.data)], dtype=np.float32))
