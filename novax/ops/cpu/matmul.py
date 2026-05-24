import numpy as np

def matmul(a, b):
    from novax.core import Tensor
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError(f"matmul requires 2D tensors, got {a.shape} and {b.shape}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Shape mismatch for matmul: {a.shape} @ {b.shape}")
    return Tensor(np.matmul(a.data, b.data))
