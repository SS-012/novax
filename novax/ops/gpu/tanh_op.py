from novax.ops.launcher import launch_kernel

def tanh(a):
    return launch_kernel(a, None, "tanh_kernel", "tanhf(a[idx])")
