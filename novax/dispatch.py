"""
Dispatch layer: routes NovaX operations to the appropriate backend.
"""

try:
    from novax.core import GPU_AVAILABLE
except Exception:
    GPU_AVAILABLE = False

DEFAULT_DEVICE = "gpu" if GPU_AVAILABLE else "cpu"

if GPU_AVAILABLE:
    from novax.ops.gpu import add as gpu_add, sub as gpu_sub, mul as gpu_mul, div as gpu_div
else:
    from novax.ops.cpu import add as cpu_add, sub as cpu_sub, mul as cpu_mul, div as cpu_div


# ------------------------------
# Dynamic dispatch wrappers
# ------------------------------

def add(a, b):
    if DEFAULT_DEVICE == "gpu" and GPU_AVAILABLE and a.on_gpu and b.on_gpu:
        return gpu_add(a, b)
    return cpu_add(a, b)

def sub(a, b):
    if DEFAULT_DEVICE == "gpu" and GPU_AVAILABLE and a.on_gpu and b.on_gpu:
        return gpu_sub(a, b)
    return cpu_sub(a, b)

def mul(a, b):
    if DEFAULT_DEVICE == "gpu" and GPU_AVAILABLE and a.on_gpu and b.on_gpu:
        return gpu_mul(a, b)
    return cpu_mul(a, b)

def div(a, b):
    if DEFAULT_DEVICE == "gpu" and GPU_AVAILABLE and a.on_gpu and b.on_gpu:
        return gpu_div(a, b)
    return cpu_div(a, b)
