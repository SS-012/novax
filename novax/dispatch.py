"""
Dispatch layer: routes NovaX operations to the appropriate backend.
"""

from novax.core import GPU_AVAILABLE

DEFAULT_DEVICE = "gpu" if GPU_AVAILABLE else "cpu"

if GPU_AVAILABLE:
    from novax.ops import gpu as backend_gpu
from novax.ops import cpu as backend_cpu


# ------------------------------
# Dynamic dispatch wrappers
# ------------------------------

def add(a, b):
    if DEFAULT_DEVICE == "gpu" and GPU_AVAILABLE and a.on_gpu and b.on_gpu:
        return backend_gpu.add(a, b)
    return backend_cpu.add(a, b)

def sub(a, b):
    if DEFAULT_DEVICE == "gpu" and GPU_AVAILABLE and a.on_gpu and b.on_gpu:
        return backend_gpu.sub(a, b)
    return backend_cpu.sub(a, b)

def mul(a, b):
    if DEFAULT_DEVICE == "gpu" and GPU_AVAILABLE and a.on_gpu and b.on_gpu:
        return backend_gpu.mul(a, b)
    return backend_cpu.mul(a, b)

def div(a, b):
    if DEFAULT_DEVICE == "gpu" and GPU_AVAILABLE and a.on_gpu and b.on_gpu:
        return backend_gpu.div(a, b)
    return backend_cpu.div(a, b)
