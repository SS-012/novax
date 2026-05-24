import numpy as np

def softmax(a):
    from novax.core import Tensor
    x = a.data
    e = np.exp(x - np.max(x))
    return Tensor(e / np.sum(e))
