from novax.ops.launcher import launch_binary_float4, launch_kernel

def add(a, b):
    out = launch_binary_float4(a, b, "add_float4_kernel", "+")
    if out is not None:
        return out
    return launch_kernel(a, b, "add_kernel", "a[idx] + b[idx]")
