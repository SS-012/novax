from novax.ops.launcher import launch_kernel

def abs(a):
    return launch_kernel(a, None, "abs_kernel", "fabsf(a[idx])")
