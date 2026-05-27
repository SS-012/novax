from novax.ops.launcher import launch_binary_float4, launch_kernel

def mul(a, b):
    out = launch_binary_float4(a, b, "mul_float4_kernel", "*")
    if out is not None:
        return out
    return launch_kernel(a, b, "mul_kernel", "a[idx] * b[idx]")
