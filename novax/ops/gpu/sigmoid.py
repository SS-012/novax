from novax.ops.launcher import launch_kernel

def sigmoid(a):
    return launch_kernel(a, None, "sigmoid_kernel", "(1.0f / (1.0f + expf(-a[idx])))")
