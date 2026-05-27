from novax.ops.launcher import launch_kernel

def exp(a):
    return launch_kernel(a, None, "exp_kernel", "__expf(a[idx])")
