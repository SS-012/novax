import numpy as np

def abs(a):
    from novax.core import Tensor
    return Tensor(np.abs(a.data))
