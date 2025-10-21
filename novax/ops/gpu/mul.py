from novax.ops.launcher import launch_kernel

def mul(a, b):
    return launch_kernel(a, b, "mul_kernel", "a[idx] * b[idx]")