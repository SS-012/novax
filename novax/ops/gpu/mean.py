from novax.ops.launcher import launch_reduce

def mean(a):
    return launch_reduce(a, "sum_reduce_mean", "sum", scale=1.0 / float(a.size))
