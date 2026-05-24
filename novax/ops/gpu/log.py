from novax.ops.launcher import launch_kernel

def log(a):
    return launch_kernel(a, None, "log_kernel", "logf(a[idx])")
