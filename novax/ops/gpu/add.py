from novax.ops.launcher import launch_kernel

def add(a, b):
    return launch_kernel(a, b, "add_kernel", "a[idx] + b[idx]")