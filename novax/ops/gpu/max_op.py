from novax.ops.launcher import launch_reduce

def max(a):
    return launch_reduce(a, "max_reduce", "max")
