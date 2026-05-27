from novax.ops.launcher import launch_kernel, launch_unary_float4

def relu(a):
    out = launch_unary_float4(a, "relu_float4_kernel", "fmaxf(0.0f, av.{c})")
    if out is not None:
        return out
    return launch_kernel(a, None, "relu_kernel", "fmaxf(0.0f, a[idx])")
