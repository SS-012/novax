import numpy as np

def exp(a):
    from novax.core import Tensor
    return Tensor(np.exp(a.data))
