from novax.ops.launcher import launch_kernel

def div(a, b):
    return launch_kernel(a, b, "div_kernel", "a[idx] / b[idx]")