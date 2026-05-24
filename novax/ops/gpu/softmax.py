from novax.ops.launcher import launch_reduce, launch_kernel

def softmax(a):
    # Step 1: max for numerical stability
    max_val = launch_reduce(a, "softmax_max_reduce", "max")
    # Step 2: x - max (broadcast scalar subtraction)
    shifted = launch_kernel(a, max_val, "softmax_shift_kernel", "a[idx] - b[0]")
    # Step 3: exp(x - max)
    exp_x = launch_kernel(shifted, None, "softmax_exp_kernel", "expf(a[idx])")
    # Step 4: sum of exp
    exp_sum = launch_reduce(exp_x, "softmax_sum_reduce", "sum")
    # Step 5: divide by sum
    return launch_kernel(exp_x, exp_sum, "softmax_div_kernel", "a[idx] / b[0]")
