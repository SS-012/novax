import numpy as np

def relu(a):
    from novax.core import Tensor
    return Tensor(np.maximum(0.0, a.data))
