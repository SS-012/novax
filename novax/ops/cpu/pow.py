import numpy as np

def pow(a, b):
    from novax.core import Tensor
    return Tensor(np.power(a.data, b.data))
