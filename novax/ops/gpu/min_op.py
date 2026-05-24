from novax.ops.launcher import launch_reduce

def min(a):
    return launch_reduce(a, "min_reduce", "min")
