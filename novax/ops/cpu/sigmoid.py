import numpy as np

def sigmoid(a):
    from novax.core import Tensor
    return Tensor(1.0 / (1.0 + np.exp(-a.data)))
