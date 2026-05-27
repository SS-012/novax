from novax.ops.launcher import launch_kernel, launch_unary_float4

def neg(a):
    out = launch_unary_float4(a, "neg_float4_kernel", "(-av.{c})")
    if out is not None:
        return out
    return launch_kernel(a, None, "neg_kernel", "(-a[idx])")
