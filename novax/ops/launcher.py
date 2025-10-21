import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from novax.core import Tensor
from novax.utils import mempool

# Cache compiled kernels to avoid recompilation
_kernel_cache = {}

def get_kernel(name: str, src: str):
    """
    Compile (if needed) and return a cached CUDA kernel by name.
    """
    if name in _kernel_cache:
        return _kernel_cache[name]

    mod = SourceModule(src)
    func = mod.get_function(name)
    _kernel_cache[name] = func
    return func


def launch_kernel(a: Tensor, b: Tensor = None, op_name: str = "custom_kernel", expr: str = None):
    """
    Generic GPU kernel launcher for elementwise operations.

    Args:
        a (Tensor): Input tensor on GPU.
        b (Tensor, optional): Second input tensor for binary ops. Defaults to None.
        op_name (str): Kernel name.
        expr (str): CUDA expression to execute per element.
                     Use 'a[idx]' and optionally 'b[idx]' in the expression.
    """
    assert a.on_gpu, "Input tensor 'a' must be on GPU."
    n = a.size
    nbytes = n * 4

    # Basic sanity check for binary ops
    if b is not None:
        assert b.on_gpu, "Input tensor 'b' must be on GPU."
        assert a.size == b.size, "Tensor size mismatch"

    # Handle unary vs binary kernels
    if b is None:
        kernel_src = f"""
        __global__ void {op_name}(const float* a, float* out, int n) {{
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
    if b is None:
        func(a.gpu_ptr, out_gpu, np.int32(n), block=block, grid=grid)
    else:
        func(a.gpu_ptr, b.gpu_ptr, out_gpu, np.int32(n), block=block, grid=grid)

    return Tensor(out_gpu, gpu=True, shape=a.shape)
