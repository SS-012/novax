from novax.ops.launcher import launch_kernel

def neg(a):
    return launch_kernel(a, None, "neg_kernel", "(-a[idx])")
