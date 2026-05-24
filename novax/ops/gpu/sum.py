from novax.ops.launcher import launch_reduce

def sum(a):
    return launch_reduce(a, "sum_reduce", "sum")
