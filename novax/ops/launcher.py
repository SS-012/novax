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

_kernel_cache = {}
_stream = None
_CAPTURE_STREAM = None   # non-None during CUDAGraph capture


def _get_stream():
    """Return the active CUDA stream (capture stream takes priority)."""
    global _stream
    if _CAPTURE_STREAM is not None:
        return _CAPTURE_STREAM
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


def _optimal_block_size() -> int:
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


# ---------------------------------------------------------------------------
# Item 5: CUDA Graphs
# ---------------------------------------------------------------------------

class CUDAGraph:
    """
    Captures a sequence of NovaX GPU kernel launches into a CUDA Graph,
    then replays the whole sequence with near-zero CPU overhead.

    Usage::

        graph = nx.CUDAGraph()
        with graph:
            out = nx.relu(nx.matmul(x, W) + b).eval()   # captured, not timed

        for step in range(10_000):
            graph.replay()   # replays in ~microseconds
    """

    def __init__(self):
        self._graph_exec = None
        self._cap_stream = None

    def __enter__(self):
        global _CAPTURE_STREAM
        if cuda is None:
            raise RuntimeError("CUDAGraph requires PyCUDA (GPU not available)")
        self._cap_stream = cuda.Stream()
        _CAPTURE_STREAM = self._cap_stream
        # begin_capture: try with mode arg first (CUDA 10+), then no-arg
        try:
            mode = 0   # cudaStreamCaptureModeRelaxed = 0
            self._cap_stream.begin_capture(mode)
        except TypeError:
            try:
                self._cap_stream.begin_capture()
            except Exception as e:
                _CAPTURE_STREAM = None
                raise RuntimeError(
                    f"CUDAGraph capture unavailable (needs PyCUDA 2021.1+ / CUDA 10+): {e}"
                )
        except Exception as e:
            _CAPTURE_STREAM = None
            raise RuntimeError(f"CUDAGraph capture failed: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CAPTURE_STREAM
        _CAPTURE_STREAM = None
        if exc_type is not None:
            return False
        try:
            graph = self._cap_stream.end_capture()
            self._graph_exec = graph.instantiate()
        except Exception as e:
            raise RuntimeError(f"CUDAGraph instantiation failed: {e}")

    def replay(self):
        """Re-execute the captured graph. Near-zero Python and driver overhead."""
        if self._graph_exec is None:
            raise RuntimeError("No graph captured. Use 'with graph:' block first.")
        self._graph_exec.launch(self._cap_stream)
        self._cap_stream.synchronize()


# ---------------------------------------------------------------------------
# Item 6: fp16-aware kernel source generator
# ---------------------------------------------------------------------------

def _make_kernel_src(op_name: str, expr: str, dtype, has_b: bool, has_scalar: bool) -> str:
    """Return CUDA C kernel source for the given dtype (float32 or float16)."""
    if dtype == np.float16:
        hdr = "#include <cuda_fp16.h>\n"
        # Rewrite expr to use float intermediates so math functions work
        expr_f = (expr
                  .replace("a[idx]", "a_f")
                  .replace("b[idx]", "b_f"))
        if not has_b and not has_scalar:
            return f"""{hdr}
__global__ void {op_name}(const __half* a, __half* out, int n) {{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {{
        float a_f = __half2float(a[idx]);
        out[idx] = __float2half((float)({expr_f}));
    }}
}}"""
        elif not has_b and has_scalar:
            return f"""{hdr}
__global__ void {op_name}(const __half* a, const float s, __half* out, int n) {{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {{
        float a_f = __half2float(a[idx]);
        out[idx] = __float2half((float)({expr_f}));
    }}
}}"""
        else:
            # Binary fp16×fp16: load both as float, compute, store fp16
            return f"""{hdr}
__global__ void {op_name}(const __half* a, const __half* b, __half* out, int n) {{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {{
        float a_f = __half2float(a[idx]);
        float b_f = __half2float(b[idx]);
        out[idx] = __float2half((float)({expr_f}));
    }}
}}"""
    else:
        # float32 (original code paths)
        if not has_b and not has_scalar:
            return f"""
__global__ void {op_name}(const float* a, float* out, int n) {{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {{
        out[idx] = {expr};
    }}
}}"""
        elif not has_b and has_scalar:
            return f"""
__global__ void {op_name}(const float* a, const float s, float* out, int n) {{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {{
        out[idx] = {expr};
    }}
}}"""
        else:
            return f"""
__global__ void {op_name}(const float* a, const float* b, float* out, int n) {{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {{
        out[idx] = {expr};
    }}
}}"""


def launch_kernel(a, b=None, op_name: str = "custom_kernel", expr: str = None,
                  scalar: float | None = None):
    """
    Generic GPU kernel launcher for elementwise operations.
    Supports float32 (default) and float16 tensors.
    """
    if cuda is None:
        raise RuntimeError("GPU not available: cannot launch kernels")
    assert a.on_gpu, "Input tensor 'a' must be on GPU."
    n = a.size
    dtype = getattr(a, 'dtype', np.float32)
    elem_size = 2 if dtype == np.float16 else 4
    nbytes = n * elem_size

    if b is not None:
        assert b.on_gpu, "Input tensor 'b' must be on GPU."
        b_dtype = getattr(b, 'dtype', np.float32)
        # Mixed precision: if a is fp16 but b is fp32 (e.g. reduction scalar),
        # the expr uses b[0] — keep it as float* in the kernel.
        if dtype == np.float16 and b_dtype != np.float16:
            # Generate a mixed-mode fp16/fp32 kernel for the a[idx] OP b[0] pattern
            hdr = "#include <cuda_fp16.h>\n"
            expr_f = expr.replace("a[idx]", "a_f")   # b[0] stays as float
            kernel_src = f"""{hdr}
__global__ void {op_name}(const __half* a, const float* b, __half* out, int n) {{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {{
        float a_f = __half2float(a[idx]);
        out[idx] = __float2half((float)({expr_f}));
    }}
}}"""
        else:
            kernel_src = _make_kernel_src(op_name, expr, dtype, True, False)
    else:
        kernel_src = _make_kernel_src(op_name, expr, dtype, False, scalar is not None)

    func = get_kernel(op_name, kernel_src)
    out_gpu = mempool.alloc(nbytes)

    bs = _optimal_block_size()
    block = (bs, 1, 1)
    grid = ((n + bs - 1) // bs, 1, 1)
    stream = _get_stream()

    if b is None and scalar is None:
        func(a.gpu_ptr, out_gpu, np.int32(n), block=block, grid=grid, stream=stream)
    elif b is None and scalar is not None:
        func(a.gpu_ptr, np.float32(scalar), out_gpu, np.int32(n),
             block=block, grid=grid, stream=stream)
    else:
        func(a.gpu_ptr, b.gpu_ptr, out_gpu, np.int32(n),
             block=block, grid=grid, stream=stream)

    from importlib import import_module
    Tensor = getattr(import_module("novax.core"), "Tensor")
    result = Tensor(out_gpu, gpu=True, inputs=[a, b] if b else [a])
    result.dtype = dtype
    return result


def launch_fused(inputs, expr: str, op_name: str = "fused_kernel"):
    """Fused elementwise kernel with arbitrary number of float32 inputs."""
    if cuda is None:
        raise RuntimeError("GPU not available: cannot launch kernels")
    assert all(t.on_gpu for t in inputs), "All inputs must be on GPU"
    n = inputs[0].size
    for t in inputs:
        assert t.size == n, "All inputs must have same size"

    params = ", ".join(
        [f"const float* x{i}" for i in range(len(inputs))] + ["float* out", "int n"]
    )
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
    from importlib import import_module
    Tensor = getattr(import_module("novax.core"), "Tensor")
    return Tensor(out_gpu, gpu=True, inputs=inputs)


def launch_reduce(a, op_name: str, reduce_type: str):
    """Two-pass parallel reduction. Always returns a float32 scalar tensor."""
    if cuda is None:
        raise RuntimeError("GPU not available: cannot launch kernels")
    assert a.on_gpu, "Input tensor must be on GPU"

    # If input is fp16, upcast to float32 first via a conversion kernel
    if getattr(a, 'dtype', np.float32) == np.float16:
        a = _fp16_to_fp32(a)

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

    BS = 256
    kernel_src = f"""
    __global__ void {op_name}(const float* in, float* out, int n) {{
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
        if (tid == 0) out[blockIdx.x] = smem[0];
    }}
    """

    n = a.size
    grid_size = (n + BS - 1) // BS
    partial_gpu = mempool.alloc(grid_size * 4)
    func = get_kernel(op_name, kernel_src)
    stream = _get_stream()
    func(a.gpu_ptr, partial_gpu, np.int32(n),
         block=(BS, 1, 1), grid=(grid_size, 1, 1),
         shared=BS * 4, stream=stream)

    if grid_size == 1:
        final_gpu = partial_gpu
    else:
        final_gpu = mempool.alloc(4)
        func2_name = op_name + "_pass2"
        func2_src = kernel_src.replace(op_name, func2_name)
        func2 = get_kernel(func2_name, func2_src)
        grid2 = (grid_size + BS - 1) // BS
        partial2_gpu = mempool.alloc(grid2 * 4)
        func2(partial_gpu, partial2_gpu, np.int32(grid_size),
              block=(BS, 1, 1), grid=(grid2, 1, 1),
              shared=BS * 4, stream=stream)
        if grid2 == 1:
            final_gpu = partial2_gpu
        else:
            func3_name = op_name + "_pass3"
            func3_src = kernel_src.replace(op_name, func3_name)
            func3 = get_kernel(func3_name, func3_src)
            final_gpu = mempool.alloc(4)
            func3(partial2_gpu, final_gpu, np.int32(grid2),
                  block=(BS, 1, 1), grid=(1, 1, 1),
                  shared=BS * 4, stream=stream)

    from importlib import import_module
    Tensor = getattr(import_module("novax.core"), "Tensor")
    result = Tensor(final_gpu, gpu=True, inputs=[a])
    result.shape = (1,)
    result.size = 1
    result.dtype = np.float32
    return result


def _fp16_to_fp32(a):
    """Convert an fp16 GPU tensor to fp32 via a simple cast kernel."""
    n = a.size
    kernel_src = """
#include <cuda_fp16.h>
__global__ void fp16_to_fp32(const __half* in, float* out, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) out[idx] = __half2float(in[idx]);
}
"""
    func = get_kernel("fp16_to_fp32", kernel_src)
    out_gpu = mempool.alloc(n * 4)
    bs = _optimal_block_size()
    func(a.gpu_ptr, out_gpu, np.int32(n),
         block=(bs, 1, 1), grid=((n + bs - 1) // bs, 1, 1),
         stream=_get_stream())
    from importlib import import_module
    Tensor = getattr(import_module("novax.core"), "Tensor")
    result = Tensor(out_gpu, gpu=True, inputs=[a])
    result.shape = a.shape
    result.size = n
    result.dtype = np.float32
    return result


# ---------------------------------------------------------------------------
# Item 0: cuBLAS-backed matmul
# ---------------------------------------------------------------------------

def _launch_matmul_cublas(a, b):
    """
    Matrix multiplication via cuBLAS cublasSgemm (scikit-cuda).
    Returns None if scikit-cuda is not installed.
    """
    try:
        from skcuda import cublas as sk_cublas
    except ImportError:
        return None

    M, K = a.shape
    _, N = b.shape
    try:
        handle = sk_cublas.cublasCreate()
        out_gpu = mempool.alloc(M * N * 4)
        # cuBLAS uses column-major storage.
        # To compute C = A @ B (row-major), we compute C^T = B^T @ A^T
        # which in column-major notation is: cublasSgemm(N, M, K, B, N, A, K, C, N)
        sk_cublas.cublasSgemm(
            handle,
            'n', 'n',      # no transpose for either operand (in col-major view)
            N, M, K,
            np.float32(1.0),
            b.gpu_ptr, N,
            a.gpu_ptr, K,
            np.float32(0.0),
            out_gpu, N,
        )
        sk_cublas.cublasDestroy(handle)

        from importlib import import_module
        Tensor = getattr(import_module("novax.core"), "Tensor")
        result = Tensor(out_gpu, gpu=True, inputs=[a, b])
        result.shape = (M, N)
        result.size = M * N
        result.dtype = np.float32
        return result
    except Exception:
        return None


def _launch_matmul_tiled(a, b):
    """Tiled shared-memory matmul kernel (fallback when cuBLAS unavailable)."""
    M, K = a.shape
    K2, N = b.shape

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

    from importlib import import_module
    Tensor = getattr(import_module("novax.core"), "Tensor")
    result = Tensor(out_gpu, gpu=True, inputs=[a, b])
    result.shape = (M, N)
    result.size = M * N
    result.dtype = np.float32
    return result


def launch_matmul(a, b):
    """
    Matrix multiplication (M×K) @ (K×N) → (M×N).
    Uses cuBLAS (via scikit-cuda) when available; falls back to tiled kernel.
    """
    if cuda is None:
        raise RuntimeError("GPU not available: cannot launch kernels")
    assert a.on_gpu and b.on_gpu, "Both tensors must be on GPU"
    assert len(a.shape) == 2 and len(b.shape) == 2, "matmul requires 2D tensors"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Shape mismatch for matmul: {a.shape} @ {b.shape}"

    result = _launch_matmul_cublas(a, b)
    if result is not None:
        return result
    return _launch_matmul_tiled(a, b)


# ---------------------------------------------------------------------------
# Item 8: Fused matmul+bias (without relu) — used by auto-pattern matching
# ---------------------------------------------------------------------------

def launch_matmul_bias(x, w, bias):
    """
    Fused matmul + bias add in a single tiled kernel pass.
    x: (M, K), w: (K, N), bias: (N,) → result: (M, N)
    """
    if cuda is None:
        raise RuntimeError("GPU not available: cannot launch kernels")
    assert x.on_gpu and w.on_gpu and bias.on_gpu, "All tensors must be on GPU"
    M, K = x.shape
    K2, N = w.shape
    assert K == K2, f"Shape mismatch: {x.shape} @ {w.shape}"
    assert bias.size == N, f"Bias size {bias.size} must match output columns {N}"

    TILE = 16
    kernel_src = f"""
    #define TILE_SIZE {TILE}
    __global__ void matmul_bias_kernel(
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
        if (row < M && col < N)
            C[row * N + col] = acc + B[col];
    }}
    """
    func = get_kernel("matmul_bias_kernel", kernel_src)
    out_gpu = mempool.alloc(M * N * 4)
    block = (TILE, TILE, 1)
    grid = ((N + TILE - 1) // TILE, (M + TILE - 1) // TILE, 1)
    stream = _get_stream()
    func(x.gpu_ptr, w.gpu_ptr, bias.gpu_ptr, out_gpu,
         np.int32(M), np.int32(K), np.int32(N),
         block=block, grid=grid, stream=stream)

    from importlib import import_module
    Tensor = getattr(import_module("novax.core"), "Tensor")
    result = Tensor(out_gpu, gpu=True, inputs=[x, w, bias])
    result.shape = (M, N)
    result.size = M * N
    result.dtype = np.float32
    return result


def launch_matmul_bias_relu(x, w, bias):
    """
    Fused matmul + bias add + ReLU in a single CUDA kernel pass.
    x: (M, K), w: (K, N), bias: (N,) → result: (M, N)
    """
    if cuda is None:
        raise RuntimeError("GPU not available: cannot launch kernels")
    assert x.on_gpu and w.on_gpu and bias.on_gpu, "All tensors must be on GPU"
    M, K = x.shape
    K2, N = w.shape
    assert K == K2, f"Shape mismatch: {x.shape} @ {w.shape}"
    assert bias.size == N, f"Bias size {bias.size} must match output columns {N}"

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

    from importlib import import_module
    Tensor = getattr(import_module("novax.core"), "Tensor")
    result = Tensor(out_gpu, gpu=True, inputs=[x, w, bias])
    result.shape = (M, N)
    result.size = M * N
    result.dtype = np.float32
    return result
