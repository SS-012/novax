import numpy as np

def tanh(a):
    from novax.core import Tensor
    return Tensor(np.tanh(a.data))
