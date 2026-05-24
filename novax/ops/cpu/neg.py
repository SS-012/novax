import numpy as np

def neg(a):
    from novax.core import Tensor
    return Tensor(-a.data)
