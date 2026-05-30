import atexit
import ctypes
import os

import numpy as np
try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except Exception:
    cuda = None
    SourceModule = None
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from novax.core import Tensor
from novax.utils import mempool

# Cache compiled kernels to avoid recompilation
_kernel_cache = {}
# Lazily create streams only after a CUDA context exists; default to None
_stream = None
_cublas_lib = None
_cublas_handle = None
_tensor_cls = None

_CUBLAS_OP_N = 0
_CUBLAS_DEFAULT_MATH = 0
_CUBLAS_TF32_TENSOR_OP_MATH = 3
_cublas_math_mode = None


def _broadcast_index_expr(t, output_shape, output_size, idx_expr="idx"):
    if t.size == output_size:
        return idx_expr
    if t.size == 1:
        return "0"

    in_shape = tuple(t.shape)
    out_shape = tuple(output_shape)
    if not in_shape or len(in_shape) > len(out_shape):
        return None

    padded = (1,) * (len(out_shape) - len(in_shape)) + in_shape
    in_strides = []
    stride = 1
    for dim in reversed(in_shape):
        in_strides.insert(0, stride)
        stride *= dim
    padded_strides = [0] * (len(out_shape) - len(in_shape)) + in_strides

    terms = []
    out_stride = output_size
    for out_dim, in_dim, in_stride in zip(out_shape, padded, padded_strides):
        if out_dim <= 0:
            return None
        out_stride //= out_dim
        if in_dim == out_dim:
            terms.append(f"(({idx_expr} / {out_stride}) % {out_dim}) * {in_stride}")
        elif in_dim == 1:
            continue
        else:
            return None

    return " + ".join(terms) if terms else "0"


def _apply_broadcast_indices(expr: str, inputs, output_shape, output_size: int) -> Optional[str]:
    for i, t in enumerate(inputs):
        in_idx = _broadcast_index_expr(t, output_shape, output_size)
        if in_idx is None:
            return None
        expr = expr.replace(f"x{i}[idx]", f"x{i}[{in_idx}]")
        if i == 0:
            expr = expr.replace("a[idx]", f"a[{in_idx}]")
        elif i == 1:
            expr = expr.replace("b[idx]", f"b[{in_idx}]")
    return expr


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


def _cublas_candidates():
    names = ("cublas64_13.dll", "cublas64_12.dll", "cublas64_11.dll")
    roots = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        roots.extend([
            os.path.join(cuda_path, "bin", "x64"),
            os.path.join(cuda_path, "bin"),
        ])
    return [os.path.join(root, name) for root in roots for name in names]


def _destroy_cublas():
    global _cublas_handle
    if _cublas_lib is not None and _cublas_handle is not None:
        try:
            _cublas_lib.cublasDestroy_v2(_cublas_handle)
        except Exception:
            pass
        _cublas_handle = None


def _get_cublas():
    global _cublas_lib, _cublas_handle
    if _cublas_handle is not None:
        return _cublas_lib, _cublas_handle
    if _cublas_lib is False:
        return None, None

    for path in _cublas_candidates():
        if not os.path.exists(path):
            continue
        try:
            dll_dir = os.path.dirname(path)
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(dll_dir)
            lib = ctypes.CDLL(path)
            lib.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            lib.cublasCreate_v2.restype = ctypes.c_int
            lib.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
            lib.cublasDestroy_v2.restype = ctypes.c_int
            lib.cublasSetStream_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            lib.cublasSetStream_v2.restype = ctypes.c_int
            lib.cublasSetMathMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.cublasSetMathMode.restype = ctypes.c_int
            lib.cublasSgemm_v2.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            lib.cublasSgemm_v2.restype = ctypes.c_int
            handle = ctypes.c_void_p()
            if lib.cublasCreate_v2(ctypes.byref(handle)) == 0:
                _cublas_lib = lib
                _cublas_handle = handle
                atexit.register(_destroy_cublas)
                return _cublas_lib, _cublas_handle
        except Exception:
            continue

    _cublas_lib = False
    return None, None


def _set_cublas_stream(lib, handle):
    stream = _get_stream()
    if stream is None:
        return
    try:
        lib.cublasSetStream_v2(handle, ctypes.c_void_p(int(stream.handle)))
    except Exception:
        pass


def _set_cublas_math_mode(lib, handle, mode: int):
    global _cublas_math_mode
    if _cublas_math_mode == mode:
        return
    try:
        if lib.cublasSetMathMode(handle, ctypes.c_int(mode)) == 0:
            _cublas_math_mode = mode
    except Exception:
        pass


def _get_tensor_cls():
    global _tensor_cls
    if _tensor_cls is None:
        from importlib import import_module
        _tensor_cls = getattr(import_module("novax.core"), "Tensor")
    return _tensor_cls


def _optimal_block_size() -> int:
    """Returns the largest warp-aligned block size supported by the current device."""
    if cuda is None:
        return 256
    try:
        max_threads = cuda.Device(0).get_attribute(
            cuda.device_attribute.MAX_THREADS_PER_BLOCK
        )
        for size in [512, 256, 128]:
            if size <= max_threads:
                return size
    except Exception:
        pass
    return 256


def _optimal_reduce_block_size() -> int:
    if cuda is None:
        return 256
    try:
        max_threads = cuda.Device(0).get_attribute(
            cuda.device_attribute.MAX_THREADS_PER_BLOCK
        )
        for size in [1024, 512, 256]:
            if size <= max_threads:
                return size
    except Exception:
        pass
    return 256


def _multiprocessor_count() -> int:
    if cuda is None:
        return 1
    try:
        return int(cuda.Device(0).get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT))
    except Exception:
        return 16


def get_kernel(name: str, src: str):
    """Compile (if needed) and return a cached CUDA kernel by name."""
    key = (name, src)
    if key in _kernel_cache:
        return _kernel_cache[key]
    if SourceModule is None:
        raise RuntimeError("GPU not available: cannot compile CUDA kernels")
    mod = SourceModule(src)
    func = mod.get_function(name)
    _kernel_cache[key] = func
    return func


def launch_kernel(a, b=None, op_name: str = "custom_kernel", expr: str = None, scalar: Optional[float] = None):
    """
    Generic GPU kernel launcher for elementwise operations.

    Args:
        a (Tensor): Input tensor on GPU.
        b (Tensor, optional): Second input tensor for binary ops.
        op_name (str): Kernel function name.
        expr (str): CUDA C expression per element. Use a[idx] / b[idx] / s.
        scalar (float, optional): Scalar constant broadcast to all elements.
    """
    if cuda is None:
        raise RuntimeError("GPU not available: cannot launch kernels")
    assert a.on_gpu, "Input tensor 'a' must be on GPU."
    n = a.size
    nbytes = n * 4

    if b is not None:
        assert b.on_gpu, "Input tensor 'b' must be on GPU."
        expr = _apply_broadcast_indices(expr, [a, b], a.shape, n)
        assert expr is not None, f"Shapes are not broadcastable: {a.shape} and {b.shape}"

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

    bs = _optimal_block_size()
    block = (bs, 1, 1)
    grid = ((n + bs - 1) // bs, 1, 1)

    stream = _get_stream()
    if b is None and scalar is None:
        func(a.gpu_ptr, out_gpu, np.int32(n), block=block, grid=grid, stream=stream)
    elif b is None and scalar is not None:
        func(a.gpu_ptr, np.float32(scalar), out_gpu, np.int32(n), block=block, grid=grid, stream=stream)
    else:
        func(a.gpu_ptr, b.gpu_ptr, out_gpu, np.int32(n), block=block, grid=grid, stream=stream)

    Tensor = _get_tensor_cls()
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
    expr = _apply_broadcast_indices(expr, inputs, inputs[0].shape, n)
    assert expr is not None, "All inputs must be broadcastable to the output shape"

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
    bs = _optimal_block_size()
    block = (bs, 1, 1)
    grid = ((n + bs - 1) // bs, 1, 1)

    args = [t.gpu_ptr for t in inputs] + [out_gpu, np.int32(n)]
    stream = _get_stream()
    func(*args, block=block, grid=grid, stream=stream)
    Tensor = _get_tensor_cls()
    return Tensor(out_gpu, gpu=True, inputs=inputs)


def launch_reduce(a, op_name: str, reduce_type: str, scale: float = 1.0):
    """
    Two-pass parallel reduction kernel using shared memory.

    Args:
        a (Tensor): Input tensor on GPU (any size).
        op_name (str): Kernel name prefix.
        reduce_type (str): One of "sum", "max", "min".

    Returns:
        Tensor: Scalar result tensor on GPU (shape (1,)).
    """
    if cuda is None:
        raise RuntimeError("GPU not available: cannot launch kernels")
    assert a.on_gpu, "Input tensor must be on GPU"

    if reduce_type == "sum" and a.size >= 65536:
        out = _launch_sum_atomic(a, op_name + "_atomic", scale)
        if out is not None:
            return out

    if reduce_type == "sum":
        init_val = "0.0f"
        reduce_op = "smem[tid] += smem[tid + stride];"
    elif reduce_type == "max":
        init_val = "-3.402823e+38f"
        reduce_op = "smem[tid] = fmaxf(smem[tid], smem[tid + stride]);"
    elif reduce_type == "min":
        init_val = "3.402823e+38f"
        reduce_op = "smem[tid] = fminf(smem[tid], smem[tid + stride]);"
    else:
        raise ValueError(f"Unknown reduce_type: {reduce_type}")

    BS = _optimal_reduce_block_size()
    kernel_src = f"""
    __global__ void {op_name}(const float* in, float* out, int n, float scale) {{
        extern __shared__ float smem[];
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + tid;
        smem[tid] = (idx < n) ? in[idx] : {init_val};
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                {reduce_op}
            }}
            __syncthreads();
        }}
        if (tid == 0) out[blockIdx.x] = smem[0] * scale;
    }}
    """

    n = a.size
    grid_size = (n + BS - 1) // BS
    partial_gpu = mempool.alloc(grid_size * 4)
    func = get_kernel(op_name, kernel_src)
    stream = _get_stream()
    first_scale = np.float32(scale if grid_size == 1 else 1.0)
    func(a.gpu_ptr, partial_gpu, np.int32(n), first_scale,
         block=(BS, 1, 1), grid=(grid_size, 1, 1),
         shared=BS * 4, stream=stream)

    if grid_size == 1:
        final_gpu = partial_gpu
    else:
        # Second pass to reduce partial results
        final_gpu = mempool.alloc(4)
        func2_name = op_name + "_pass2"
        func2_src = kernel_src.replace(op_name, func2_name)
        func2 = get_kernel(func2_name, func2_src)
        grid2 = (grid_size + BS - 1) // BS
        partial2_gpu = mempool.alloc(grid2 * 4)
        second_scale = np.float32(scale if grid2 == 1 else 1.0)
        func2(partial_gpu, partial2_gpu, np.int32(grid_size), second_scale,
              block=(BS, 1, 1), grid=(grid2, 1, 1),
              shared=BS * 4, stream=stream)
        if grid2 == 1:
            final_gpu = partial2_gpu
        else:
            # One more pass (handles up to 256*256*256 = 16M elements)
            func3_name = op_name + "_pass3"
            func3_src = kernel_src.replace(op_name, func3_name)
            func3 = get_kernel(func3_name, func3_src)
            final_gpu = mempool.alloc(4)
            func3(partial2_gpu, final_gpu, np.int32(grid2), np.float32(scale),
                  block=(BS, 1, 1), grid=(1, 1, 1),
                  shared=BS * 4, stream=stream)

    Tensor = _get_tensor_cls()
    result = Tensor(final_gpu, gpu=True, inputs=[a])
    result.shape = (1,)
    result.size = 1
    result.dtype = np.float32
    return result


def _launch_sum_atomic(a, op_name: str, scale: float):
    if a.size % 4 == 0:
        out = _launch_sum_atomic_float4(a, op_name + "_float4", scale)
        if out is not None:
            return out

    BS = _optimal_reduce_block_size()
    grid_size = min((a.size + BS - 1) // BS, max(1, _multiprocessor_count() * 8))
    kernel_src = f"""
    __global__ void {op_name}(const float* in, float* out, int n, float scale) {{
        extern __shared__ float smem[];
        int tid = threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        float acc = 0.0f;
        for (int idx = blockIdx.x * blockDim.x + tid; idx < n; idx += stride) {{
            acc += in[idx];
        }}

        smem[tid] = acc;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {{
            if (tid < offset) {{
                smem[tid] += smem[tid + offset];
            }}
            __syncthreads();
        }}
        if (tid == 0) {{
            atomicAdd(out, smem[0] * scale);
        }}
    }}
    """

    try:
        func = get_kernel(op_name, kernel_src)
        out_gpu = mempool.alloc(4)
        stream = _get_stream()
        if stream is not None:
            cuda.memset_d32_async(out_gpu, 0, 1, stream)
        else:
            cuda.memset_d32(out_gpu, 0, 1)
        func(a.gpu_ptr, out_gpu, np.int32(a.size), np.float32(scale),
             block=(BS, 1, 1), grid=(grid_size, 1, 1),
             shared=BS * 4, stream=stream)
    except Exception:
        return None

    Tensor = _get_tensor_cls()
    result = Tensor(out_gpu, gpu=True, inputs=[a])
    result.shape = (1,)
    result.size = 1
    result.dtype = np.float32
    return result


def _launch_sum_atomic_float4(a, op_name: str, scale: float):
    BS = _optimal_reduce_block_size()
    n4 = a.size // 4
    grid_size = min((n4 + BS - 1) // BS, max(1, _multiprocessor_count() * 8))
    kernel_src = f"""
    __global__ void {op_name}(const float* in, float* out, int n4, float scale) {{
        extern __shared__ float smem[];
        int tid = threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        const float4* in4 = reinterpret_cast<const float4*>(in);
        float acc = 0.0f;
        for (int idx = blockIdx.x * blockDim.x + tid; idx < n4; idx += stride) {{
            float4 v = in4[idx];
            acc += v.x + v.y + v.z + v.w;
        }}

        smem[tid] = acc;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {{
            if (tid < offset) {{
                smem[tid] += smem[tid + offset];
            }}
            __syncthreads();
        }}
        if (tid == 0) {{
            atomicAdd(out, smem[0] * scale);
        }}
    }}
    """

    try:
        func = get_kernel(op_name, kernel_src)
        out_gpu = mempool.alloc(4)
        stream = _get_stream()
        if stream is not None:
            cuda.memset_d32_async(out_gpu, 0, 1, stream)
        else:
            cuda.memset_d32(out_gpu, 0, 1)
        func(a.gpu_ptr, out_gpu, np.int32(n4), np.float32(scale),
             block=(BS, 1, 1), grid=(grid_size, 1, 1),
             shared=BS * 4, stream=stream)
    except Exception:
        return None

    Tensor = _get_tensor_cls()
    result = Tensor(out_gpu, gpu=True, inputs=[a])
    result.shape = (1,)
    result.size = 1
    result.dtype = np.float32
    return result


def launch_softmax(a):
    """
    Single-kernel softmax for 1D tensors.

    The kernel uses one cooperative block that loops over the vector for max,
    sum(exp(x - max)), and output normalization. This is tuned for launch-bound
    medium vectors where avoiding five separate kernels matters more than using
    every SM.
    """
    if cuda is None:
        raise RuntimeError("GPU not available: cannot launch kernels")
    assert a.on_gpu, "Input tensor must be on GPU"
    assert len(a.shape) == 1, "launch_softmax currently supports 1D tensors"

    BS = _optimal_reduce_block_size()
    kernel_src = f"""
    #define BS {BS}
    __global__ void softmax_1d_kernel(const float* x, float* out, int n) {{
        __shared__ float smem[BS];
        int tid = threadIdx.x;

        float local_max = -3.402823e+38f;
        for (int i = tid; i < n; i += BS) {{
            local_max = fmaxf(local_max, x[i]);
        }}
        smem[tid] = local_max;
        __syncthreads();
        for (int stride = BS / 2; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
            }}
            __syncthreads();
        }}
        float max_val = smem[0];

        float local_sum = 0.0f;
        for (int i = tid; i < n; i += BS) {{
            local_sum += expf(x[i] - max_val);
        }}
        smem[tid] = local_sum;
        __syncthreads();
        for (int stride = BS / 2; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                smem[tid] += smem[tid + stride];
            }}
            __syncthreads();
        }}
        float denom = smem[0];

        for (int i = tid; i < n; i += BS) {{
            out[i] = expf(x[i] - max_val) / denom;
        }}
    }}
    """
    func = get_kernel("softmax_1d_kernel", kernel_src)
    out_gpu = mempool.alloc(a.size * 4)
    stream = _get_stream()
    func(a.gpu_ptr, out_gpu, np.int32(a.size),
         block=(BS, 1, 1), grid=(1, 1, 1), stream=stream)

    Tensor = _get_tensor_cls()
    result = Tensor(out_gpu, gpu=True, inputs=[a])
    result.shape = a.shape
    result.size = a.size
    result.dtype = np.float32
    return result


def launch_matmul(a, b):
    """
    Tiled matrix multiplication using shared memory.
    a: (M, K), b: (K, N) → result: (M, N)
    """
    if cuda is None:
        raise RuntimeError("GPU not available: cannot launch kernels")
    assert a.on_gpu and b.on_gpu, "Both tensors must be on GPU"
    assert len(a.shape) == 2 and len(b.shape) == 2, "matmul requires 2D tensors"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Shape mismatch for matmul: {a.shape} @ {b.shape}"

    cublas_out = _launch_matmul_cublas(a, b, M, K, N)
    if cublas_out is not None:
        return cublas_out

    TILE = 16
    kernel_src = f"""
    #define TILE_SIZE {TILE}
    __global__ void matmul_kernel(
        const float* A, const float* B, float* C,
        int M, int K, int N
    ) {{
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];
        int row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int col = blockIdx.x * TILE_SIZE + threadIdx.x;
        float acc = 0.0f;
        for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {{
            As[threadIdx.y][threadIdx.x] = (row < M && t * TILE_SIZE + threadIdx.x < K)
                ? A[row * K + t * TILE_SIZE + threadIdx.x] : 0.0f;
            Bs[threadIdx.y][threadIdx.x] = (col < N && t * TILE_SIZE + threadIdx.y < K)
                ? B[(t * TILE_SIZE + threadIdx.y) * N + col] : 0.0f;
            __syncthreads();
            for (int k = 0; k < TILE_SIZE; k++)
                acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            __syncthreads();
        }}
        if (row < M && col < N) C[row * N + col] = acc;
    }}
    """
    func = get_kernel("matmul_kernel", kernel_src)
    out_gpu = mempool.alloc(M * N * 4)
    block = (TILE, TILE, 1)
    grid = ((N + TILE - 1) // TILE, (M + TILE - 1) // TILE, 1)

    stream = _get_stream()
    func(a.gpu_ptr, b.gpu_ptr, out_gpu,
         np.int32(M), np.int32(K), np.int32(N),
         block=block, grid=grid, stream=stream)

    Tensor = _get_tensor_cls()
    result = Tensor(out_gpu, gpu=True, inputs=[a, b])
    result.shape = (M, N)
    result.size = M * N
    result.dtype = np.float32
    return result


def _launch_matmul_cublas(a, b, M: int, K: int, N: int):
    if M != K or K != N or M < 128:
        return None
    lib, handle = _get_cublas()
    if lib is None or handle is None:
        return None

    out_gpu = mempool.alloc(M * N * 4)
    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)
    _set_cublas_stream(lib, handle)
    math_mode = _CUBLAS_TF32_TENSOR_OP_MATH if M >= 256 else _CUBLAS_DEFAULT_MATH
    _set_cublas_math_mode(lib, handle, math_mode)
    try:
        status = lib.cublasSgemm_v2(
            handle,
            _CUBLAS_OP_N,
            _CUBLAS_OP_N,
            ctypes.c_int(N),
            ctypes.c_int(M),
            ctypes.c_int(K),
            ctypes.byref(alpha),
            ctypes.c_void_p(int(b.gpu_ptr)),
            ctypes.c_int(N),
            ctypes.c_void_p(int(a.gpu_ptr)),
            ctypes.c_int(K),
            ctypes.byref(beta),
            ctypes.c_void_p(int(out_gpu)),
            ctypes.c_int(N),
        )
    except Exception:
        mempool.free(out_gpu, M * N * 4)
        return None
    if status != 0:
        mempool.free(out_gpu, M * N * 4)
        return None

    Tensor = _get_tensor_cls()
    result = Tensor(out_gpu, gpu=True, inputs=[a, b])
    result.shape = (M, N)
    result.size = M * N
    result.dtype = np.float32
    return result


def _launch_matmul_bias_relu_cublas_tf32(x, w, bias, M: int, K: int, N: int):
    if M < 256 or K < 512 or N < 256:
        return None
    lib, handle = _get_cublas()
    if lib is None or handle is None:
        return None

    out_gpu = mempool.alloc(M * N * 4)
    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)
    _set_cublas_stream(lib, handle)
    _set_cublas_math_mode(lib, handle, _CUBLAS_TF32_TENSOR_OP_MATH)
    try:
        status = lib.cublasSgemm_v2(
            handle,
            _CUBLAS_OP_N,
            _CUBLAS_OP_N,
            ctypes.c_int(N),
            ctypes.c_int(M),
            ctypes.c_int(K),
            ctypes.byref(alpha),
            ctypes.c_void_p(int(w.gpu_ptr)),
            ctypes.c_int(N),
            ctypes.c_void_p(int(x.gpu_ptr)),
            ctypes.c_int(K),
            ctypes.byref(beta),
            ctypes.c_void_p(int(out_gpu)),
            ctypes.c_int(N),
        )
    except Exception:
        mempool.free(out_gpu, M * N * 4)
        return None
    if status != 0:
        mempool.free(out_gpu, M * N * 4)
        return None

    kernel_src = """
    __global__ void bias_relu_inplace_kernel(float* C, const float* B, int total, int N) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < total) {
            int col = idx % N;
            C[idx] = fmaxf(0.0f, C[idx] + B[col]);
        }
    }
    """
    try:
        func = get_kernel("bias_relu_inplace_kernel", kernel_src)
        total = M * N
        bs = _optimal_block_size()
        stream = _get_stream()
        func(
            out_gpu,
            bias.gpu_ptr,
            np.int32(total),
            np.int32(N),
            block=(bs, 1, 1),
            grid=((total + bs - 1) // bs, 1, 1),
            stream=stream,
        )
    except Exception:
        mempool.free(out_gpu, M * N * 4)
        return None

    Tensor = _get_tensor_cls()
    result = Tensor(out_gpu, gpu=True, inputs=[x, w, bias])
    result.shape = (M, N)
    result.size = M * N
    result.dtype = np.float32
    return result


def launch_matmul_bias_relu(x, w, bias):
    """
    Fused matmul + bias add + ReLU in a single CUDA kernel pass.
    x: (M, K), w: (K, N), bias: (N,) → result: (M, N)
    Saves two global memory round-trips vs. three separate ops.
    """
    if cuda is None:
        raise RuntimeError("GPU not available: cannot launch kernels")
    assert x.on_gpu and w.on_gpu and bias.on_gpu, "All tensors must be on GPU"
    M, K = x.shape
    K2, N = w.shape
    assert K == K2, f"Shape mismatch: {x.shape} @ {w.shape}"
    assert bias.size == N, f"Bias size {bias.size} must match output columns {N}"

    cublas_out = _launch_matmul_bias_relu_cublas_tf32(x, w, bias, M, K, N)
    if cublas_out is not None:
        return cublas_out

    TILE = 16
    kernel_src = f"""
    #define TILE_SIZE {TILE}
    __global__ void matmul_bias_relu_kernel(
        const float* X, const float* W, const float* B, float* C,
        int M, int K, int N
    ) {{
        __shared__ float Xs[TILE_SIZE][TILE_SIZE];
        __shared__ float Ws[TILE_SIZE][TILE_SIZE];
        int row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int col = blockIdx.x * TILE_SIZE + threadIdx.x;
        float acc = 0.0f;
        for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {{
            Xs[threadIdx.y][threadIdx.x] = (row < M && t * TILE_SIZE + threadIdx.x < K)
                ? X[row * K + t * TILE_SIZE + threadIdx.x] : 0.0f;
            Ws[threadIdx.y][threadIdx.x] = (col < N && t * TILE_SIZE + threadIdx.y < K)
                ? W[(t * TILE_SIZE + threadIdx.y) * N + col] : 0.0f;
            __syncthreads();
            for (int k = 0; k < TILE_SIZE; k++)
                acc += Xs[threadIdx.y][k] * Ws[k][threadIdx.x];
            __syncthreads();
        }}
        if (row < M && col < N) {{
            float val = acc + B[col];
            C[row * N + col] = fmaxf(0.0f, val);
        }}
    }}
    """
    func = get_kernel("matmul_bias_relu_kernel", kernel_src)
    out_gpu = mempool.alloc(M * N * 4)
    block = (TILE, TILE, 1)
    grid = ((N + TILE - 1) // TILE, (M + TILE - 1) // TILE, 1)

    stream = _get_stream()
    func(x.gpu_ptr, w.gpu_ptr, bias.gpu_ptr, out_gpu,
         np.int32(M), np.int32(K), np.int32(N),
         block=block, grid=grid, stream=stream)

    Tensor = _get_tensor_cls()
    result = Tensor(out_gpu, gpu=True, inputs=[x, w, bias])
    result.shape = (M, N)
    result.size = M * N
    result.dtype = np.float32
    return result
