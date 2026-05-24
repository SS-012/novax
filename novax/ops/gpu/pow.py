from novax.ops.launcher import launch_kernel

def pow(a, b):
    return launch_kernel(a, b, "pow_kernel", "powf(a[idx], b[idx])")
