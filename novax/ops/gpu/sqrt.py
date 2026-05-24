from novax.ops.launcher import launch_kernel

def sqrt(a):
    return launch_kernel(a, None, "sqrt_kernel", "sqrtf(a[idx])")
