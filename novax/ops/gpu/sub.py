from novax.ops.launcher import launch_kernel

def sub(a, b):
    return launch_kernel(a, b, "sub_kernel", "a[idx] - b[idx]")