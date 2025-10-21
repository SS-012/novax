import numpy as np
try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except Exception:
    cuda = None
    SourceModule = None
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from novax.core import Tensor
from novax.utils import mempool

# Cache compiled kernels to avoid recompilation
_kernel_cache = {}
# Lazily create streams only after a CUDA context exists; default to None
_stream = None


def _get_stream():
    global _stream
    if cuda is None:
        return None
    try:
        ctx = cuda.Context.get_current()
    except Exception:
        ctx = None
    if ctx is None:
        return None
    if _stream is None:
        try:
            _stream = cuda.Stream()
        except Exception:
            _stream = None
    return _stream

def get_kernel(name: str, src: str):
    """
    Compile (if needed) and return a cached CUDA kernel by name.
    """
    key = (name, src)
    if key in _kernel_cache:
        return _kernel_cache[key]
    if SourceModule is None:
        raise RuntimeError("GPU not available: cannot compile CUDA kernels")
    mod = SourceModule(src)
    func = mod.get_function(name)
    _kernel_cache[key] = func
    return func


def launch_kernel(a, b = None, op_name: str = "custom_kernel", expr: str = None, scalar: float | None = None):
    """
    Generic GPU kernel launcher for elementwise operations.

    Args:
        a (Tensor): Input tensor on GPU.
        b (Tensor, optional): Second input tensor for binary ops. Defaults to None.
        op_name (str): Kernel name.
        expr (str): CUDA expression to execute per element.
                     Use 'a[idx]' and optionally 'b[idx]' in the expression.
    """
    if cuda is None:
        raise RuntimeError("GPU not available: cannot launch kernels")
    assert a.on_gpu, "Input tensor 'a' must be on GPU."
    n = a.size
    nbytes = n * 4

    # Basic sanity check for binary ops
    if b is not None:
        assert b.on_gpu, "Input tensor 'b' must be on GPU."
        assert a.size == b.size, "Tensor size mismatch"

    # Handle unary/binary/scalar kernels
    if b is None and scalar is None:
        kernel_src = f"""
        __global__ void {op_name}(const float* a, float* out, int n) {{
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < n) {{
                out[idx] = {expr};
            }}
        }}
        """
    elif b is None and scalar is not None:
        kernel_src = f"""
        __global__ void {op_name}(const float* a, const float s, float* out, int n) {{
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < n) {{
                out[idx] = {expr};
            }}
        }}
        """
    else:
        kernel_src = f"""
        __global__ void {op_name}(const float* a, const float* b, float* out, int n) {{
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < n) {{
                out[idx] = {expr};
            }}
        }}
        """

    func = get_kernel(op_name, kernel_src)
    out_gpu = mempool.alloc(nbytes)

    block = (256, 1, 1)
    grid = ((n + block[0] - 1) // block[0], 1, 1)

    # Dispatch correct kernel signature
    stream = _get_stream()
    if b is None and scalar is None:
        func(a.gpu_ptr, out_gpu, np.int32(n), block=block, grid=grid, stream=stream)
    elif b is None and scalar is not None:
        func(a.gpu_ptr, np.float32(scalar), out_gpu, np.int32(n), block=block, grid=grid, stream=stream)
    else:
        func(a.gpu_ptr, b.gpu_ptr, out_gpu, np.int32(n), block=block, grid=grid, stream=stream)

    # Lazy import to avoid circular dependency at module import time
    from importlib import import_module
    Tensor = getattr(import_module("novax.core"), "Tensor")
    return Tensor(out_gpu, gpu=True, inputs=[a, b] if b else [a])


def launch_fused(inputs, expr: str, op_name: str = "fused_kernel"):
    """
    Launch a fused elementwise kernel with arbitrary number of inputs.
    'expr' should reference x0[idx], x1[idx], ... according to inputs order.
    """
    if cuda is None:
        raise RuntimeError("GPU not available: cannot launch kernels")
    assert all(t.on_gpu for t in inputs), "All inputs must be on GPU"
    n = inputs[0].size
    for t in inputs:
        assert t.size == n, "All inputs must have same size"

    params = ", ".join([f"const float* x{i}" for i in range(len(inputs))] + ["float* out", "int n"])
    kernel_src = f"""
    __global__ void {op_name}({params}) {{
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {{
            out[idx] = {expr};
        }}
    }}
    """
    func = get_kernel(op_name, kernel_src)
    out_gpu = mempool.alloc(n * 4)
    block = (256, 1, 1)
    grid = ((n + block[0] - 1) // block[0], 1, 1)

    args = [t.gpu_ptr for t in inputs] + [out_gpu, np.int32(n)]
    stream = _get_stream()
    func(*args, block=block, grid=grid, stream=stream)
    from importlib import import_module
    Tensor = getattr(import_module("novax.core"), "Tensor")
    return Tensor(out_gpu, gpu=True, inputs=inputs)