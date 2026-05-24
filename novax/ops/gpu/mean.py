import numpy as np
from novax.ops.launcher import launch_reduce

def mean(a):
    from novax.core import Tensor
    s = launch_reduce(a, "sum_reduce_mean", "sum")
    # divide GPU result by n using a scalar kernel
    from novax.ops.launcher import launch_kernel
    return launch_kernel(s, None, "mean_div_kernel", f"a[idx] / {float(a.size)}f")
