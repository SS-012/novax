from novax.ops.launcher import launch_reduce, launch_kernel


def softmax(a):
    """
    GPU softmax. Tries the Triton fused 1-pass kernel first (Item 7);
    falls back to the 5-pass PyCUDA implementation for large n or when
    Triton is unavailable.
    """
    # Item 7: single-pass Triton kernel (n ≤ 65536)
    try:
        from novax.ops.gpu.triton_softmax import triton_softmax_1d
        result = triton_softmax_1d(a)
        if result is not None:
            return result
    except Exception:
        pass

    # Fallback: 5-pass PyCUDA implementation
    max_val = launch_reduce(a, "softmax_max_reduce", "max")
    shifted = launch_kernel(a, max_val, "softmax_shift_kernel", "a[idx] - b[0]")
    exp_x   = launch_kernel(shifted, None, "softmax_exp_kernel", "expf(a[idx])")
    exp_sum = launch_reduce(exp_x, "softmax_sum_reduce", "sum")
    return launch_kernel(exp_x, exp_sum, "softmax_div_kernel", "a[idx] / b[0]")
