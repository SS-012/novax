from novax.ops.launcher import launch_reduce, launch_kernel


def softmax(a):
    """
    GPU softmax over a flat tensor:

        m = max(a);  e = exp(a - m);  s = sum(e);  out = e / s

    Two warp-shuffle reductions (max, sum) plus two elementwise passes, each
    saturating all SMs. The max-subtract and exp are fused into one kernel.

    A previous Triton single-block implementation was removed: it ran the whole
    reduction on one SM and round-tripped the data through PyTorch tensors with
    a full device synchronize, which benchmarked ~10× slower than this
    multi-pass path for n in the tens of thousands.
    """
    max_val = launch_reduce(a, "softmax_max_reduce", "max")
    exp_x   = launch_kernel(a, max_val, "softmax_exp_shift_kernel",
                            "expf(a[idx] - b[0])")
    exp_sum = launch_reduce(exp_x, "softmax_sum_reduce", "sum")
    return launch_kernel(exp_x, exp_sum, "softmax_div_kernel", "a[idx] / b[0]")
