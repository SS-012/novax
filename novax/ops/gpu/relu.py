from novax.ops.launcher import launch_kernel

def relu(a):
    return launch_kernel(a, None, "relu_kernel", "(a[idx] > 0.0f ? a[idx] : 0.0f)")
