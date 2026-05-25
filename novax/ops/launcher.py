import re
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
_graph_replay_cache = {}   # T4: (expr_hash, n, ptr_tuple) → _GraphReplay


class _GraphReplay:
    """Cached CUDA Graph entry for a fused kernel with stable input pointers."""
    __slots__ = ("graph_exec", "cap_stream", "out_ptr", "shape", "size")

    def __init__(self, graph_exec, cap_stream, out_ptr, shape, size):
        self.graph_exec = graph_exec
        self.cap_stream = cap_stream
        self.out_ptr = out_ptr
        self.shape = shape
        self.size = size


_stream = None
_stream_initialized = False   # True once we've resolved _stream (cached)
_CAPTURE_STREAM = None         # non-None during CUDAGraph capture
_BLOCK_SIZE = None             # cached optimal block size (device query is costly)


def _get_stream():
    """Return the active CUDA stream (capture stream takes priority).

    The default stream is resolved once and cached; subsequent calls avoid the
    per-launch ``Context.get_current()`` driver round-trip.
    """
    if _CAPTURE_STREAM is not None:
        return _CAPTURE_STREAM
    global _stream, _stream_initialized
    if _stream_initialized:
        return _stream
    if cuda is None:
        return None
    try:
        ctx = cuda.Context.get_current()
    except Exception:
        ctx = None
    if ctx is None:
        return None   # no context yet → retry on a later call
    try:
        _stream = cuda.Stream()
    except Exception:
        _stream = None
    _stream_initialized = True
    return _stream


def _optimal_block_size() -> int:
    """Return the launch block size, querying the device only once."""
    global _BLOCK_SIZE
    if _BLOCK_SIZE is not None:
        return _BLOCK_SIZE
    if cuda is None:
        return 256
    try:
        max_threads = cuda.Device(0).get_attribute(
            cuda.device_attribute.MAX_THREADS_PER_BLOCK
        )
        for size in [512, 256, 128]:
            if size <= max_threads:
                _BLOCK_SIZE = size
                return size
    except Exception:
        pass
    _BLOCK_SIZE = 256
    return _BLOCK_SIZE


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


_Tensor = None


def _tensor_cls():
    """Return the Tensor class, resolved once (avoids a per-launch import)."""
    global _Tensor
    if _Tensor is None:
        from novax.core import Tensor
        _Tensor = Tensor
    return _Tensor


# ---------------------------------------------------------------------------
# Item 5: CUDA Graphs
# ---------------------------------------------------------------------------

class CUDAGraph:
    """
    Captures a whole NovaX forward pass (every kernel + cuBLAS call it issues)
    into a single CUDA Graph, then replays the entire sequence with one launch
    and zero per-op Python overhead.

    The recommended entry point is ``capture(fn)``, which warms the kernel/cuBLAS
    caches and pre-seeds the memory pool before capturing (CUDA forbids device
    allocation while a capture is in flight)::

        graph = nx.CUDAGraph()
        out = graph.capture(lambda: nx.relu(nx.matmul(x, W1) + b1).eval())

        for _ in range(10_000):
            graph.replay()          # one graph launch, no Python per op
            result = out.to_host()  # `out` buffers are overwritten in place

    If capture is unavailable (old PyCUDA / CUDA), ``capture`` records the
    function and ``replay`` transparently falls back to re-running it, so calling
    code stays correct either way.

    The low-level ``with graph:`` form is still supported for callers that manage
    pool warm-up themselves.
    """

    def __init__(self):
        self._graph_exec = None
        self._cap_stream = None
        self._fn = None          # fallback path when capture is unavailable
        self._out = None         # captured output tensor (buffers reused on replay)

    # -- low-level context-manager capture (caller must pre-warm the pool) -----

    def __enter__(self):
        global _CAPTURE_STREAM
        if cuda is None:
            raise RuntimeError("CUDAGraph requires PyCUDA (GPU not available)")
        self._cap_stream = cuda.Stream()
        _CAPTURE_STREAM = self._cap_stream
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

    # -- high-level capture: warm, pre-seed, then capture ----------------------

    def capture(self, fn):
        """
        Capture ``fn`` (a zero-arg callable that returns a concrete GPU Tensor)
        into a replayable CUDA Graph. Returns the output Tensor; its device
        buffers are overwritten in place on every ``replay()``.
        """
        if cuda is None:
            raise RuntimeError("CUDAGraph requires PyCUDA (GPU not available)")
        self._fn = fn

        # Pass 1 — warm up: compile kernels, initialise cuBLAS, and record which
        # pool buckets the computation needs (so capture-time allocs all hit).
        mempool.begin_record()
        try:
            fn()
        finally:
            buckets = mempool.end_record()
        try:
            mempool.preseed(buckets)
        except Exception:
            pass

        # Pass 2 — capture.
        global _CAPTURE_STREAM
        self._cap_stream = cuda.Stream()
        _CAPTURE_STREAM = self._cap_stream
        try:
            try:
                self._cap_stream.begin_capture(0)
            except TypeError:
                self._cap_stream.begin_capture()
            self._out = fn()
            graph = self._cap_stream.end_capture()
            self._graph_exec = graph.instantiate()
        except Exception:
            # Capture unsupported / illegal op mid-capture — fall back cleanly.
            self._graph_exec = None
            self._out = None
        finally:
            _CAPTURE_STREAM = None

        # If capture failed, run once more outside capture for a valid result.
        if self._out is None:
            self._out = fn()
        return self._out

    def replay(self):
        """
        Re-execute the captured graph (one launch, no per-op Python). Falls back
        to re-running the recorded function when no graph was captured.
        """
        if self._graph_exec is not None:
            self._graph_exec.launch(self._cap_stream)
            self._cap_stream.synchronize()
            return self._out
        if self._fn is not None:
            self._out = self._fn()
            return self._out
        raise RuntimeError("No graph captured. Call capture(fn) or use 'with graph:'.")


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

    Tensor = _tensor_cls()
    result = Tensor(out_gpu, gpu=True, inputs=[a, b] if b else [a])
    result.dtype = dtype
    return result


def _build_fused_scalar_src(op_name: str, n_inputs: int, expr: str) -> str:
    """Grid-stride scalar elementwise kernel source."""
    params = ", ".join(
        [f"const float* x{i}" for i in range(n_inputs)] + ["float* out", "int n"]
    )
    return f"""
    __global__ void {op_name}({params}) {{
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < n;
             idx += blockDim.x * gridDim.x) {{
            out[idx] = {expr};
        }}
    }}
    """


def _build_fused_v4_src(op_name: str, n_inputs: int, expr: str) -> str:
    """float4-vectorized fused kernel source (only valid when all inputs are same size)."""
    params = ", ".join(
        [f"const float* x{i}" for i in range(n_inputs)] + ["float* out", "int n4"]
    )
    loads = "\n            ".join(
        f"float4 v{i} = ((const float4*)x{i})[i];" for i in range(n_inputs)
    )
    expr_parts = []
    for c in ('x', 'y', 'z', 'w'):
        e = expr
        for i in range(n_inputs):
            e = re.sub(r'x' + str(i) + r'\[idx\]', f'v{i}.{c}', e)
        expr_parts.append(e)
    return f"""
    __global__ void {op_name}({params}) {{
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < n4;
             i += blockDim.x * gridDim.x) {{
            {loads}
            float4 vout;
            vout.x = {expr_parts[0]};
            vout.y = {expr_parts[1]};
            vout.z = {expr_parts[2]};
            vout.w = {expr_parts[3]};
            ((float4*)out)[i] = vout;
        }}
    }}
    """


def launch_fused(inputs, expr: str, op_name: str = "fused_kernel"):
    """
    Fused elementwise kernel over an arbitrary number of float32 inputs.

    Automatically:
    - Uses float4 vectorized loads when all inputs share the same size and n%4==0
      (T2: up to ~2× on memory-bound ops).
    - Captures a CUDA Graph on first call and replays on repeated calls with
      identical input pointers (T4: near-zero CPU overhead on the hot path).
    - Falls back to a plain grid-stride launch when inside a manual CUDAGraph
      capture block or when CUDA Graphs are unavailable.
    """
    if cuda is None:
        raise RuntimeError("GPU not available: cannot launch kernels")
    assert all(t.on_gpu for t in inputs), "All inputs must be on GPU"
    n = max(t.size for t in inputs)
    big = max(inputs, key=lambda t: t.size)

    # T2: float4 vectorized path — only when every input is the same full size.
    all_same = all(t.size == n for t in inputs)
    use_vec4 = (all_same and n % 4 == 0
                and all(getattr(t, 'dtype', np.float32) == np.float32 for t in inputs))
    actual_name = (op_name + "_v4") if use_vec4 else op_name

    # T4: CUDA Graph auto-capture / replay (skip if inside a manual capture).
    if _CAPTURE_STREAM is None:
        ptr_tuple = tuple(int(t.gpu_ptr) for t in inputs)
        cache_key = (hash((actual_name, expr)), n, ptr_tuple)
        entry = _graph_replay_cache.get(cache_key)
        if entry is not None:
            entry.graph_exec.launch(entry.cap_stream)
            entry.cap_stream.synchronize()
            Tensor = _tensor_cls()
            result = Tensor(entry.out_ptr, gpu=True, inputs=inputs)
            result.shape = entry.shape
            result.size = entry.size
            result.dtype = np.float32
            result.pinned = True   # memory owned by graph cache
            return result

        # First call: compile kernel, capture into a CUDA Graph, execute once.
        kernel_src = (_build_fused_v4_src(actual_name, len(inputs), expr)
                      if use_vec4
                      else _build_fused_scalar_src(actual_name, len(inputs), expr))
        func = get_kernel(actual_name, kernel_src)
        n_elems = (n // 4) if use_vec4 else n
        out_gpu = cuda.mem_alloc(n * 4)   # stable allocation outside the pool
        bs = _optimal_block_size()
        block = (bs, 1, 1)
        grid = (min((n_elems + bs - 1) // bs, 65535), 1, 1)
        args = [t.gpu_ptr for t in inputs] + [out_gpu, np.int32(n_elems)]
        try:
            cap_stream = cuda.Stream()
            cap_stream.begin_capture()
            func(*args, block=block, grid=grid, stream=cap_stream)
            graph_obj = cap_stream.end_capture()
            graph_exec = graph_obj.instantiate()
            graph_exec.launch(cap_stream)
            cap_stream.synchronize()
            _graph_replay_cache[cache_key] = _GraphReplay(
                graph_exec, cap_stream, out_gpu, big.shape, n)
            Tensor = _tensor_cls()
            result = Tensor(out_gpu, gpu=True, inputs=inputs)
            result.shape = big.shape
            result.size = n
            result.dtype = np.float32
            result.pinned = True
            return result
        except Exception:
            pass   # CUDA Graphs unavailable — fall through to normal launch

    # Normal launch: used when inside a manual CUDAGraph capture or when the
    # auto-capture above failed (e.g. PyCUDA too old / CUDA < 10).
    kernel_src = (_build_fused_v4_src(actual_name, len(inputs), expr)
                  if use_vec4
                  else _build_fused_scalar_src(actual_name, len(inputs), expr))
    func = get_kernel(actual_name, kernel_src)
    n_elems = (n // 4) if use_vec4 else n
    out_gpu = mempool.alloc(n * 4)
    bs = _optimal_block_size()
    block = (bs, 1, 1)
    grid = (min((n_elems + bs - 1) // bs, 65535), 1, 1)
    args = [t.gpu_ptr for t in inputs] + [out_gpu, np.int32(n_elems)]
    stream = _get_stream()
    func(*args, block=block, grid=grid, stream=stream)
    Tensor = _tensor_cls()
    result = Tensor(out_gpu, gpu=True, inputs=inputs)
    result.shape = big.shape
    result.size = n
    result.dtype = np.float32
    return result


def launch_broadcast_binary(big, small, op_symbol: str, small_is_left: bool,
                            op_name: str):
    """
    Trailing-dim broadcast binary op: out[idx] = a[idx] OP b[idx % m].

    `big` is the full-size operand (m·k elements), `small` the broadcast operand
    (m elements aligned to `big`'s trailing dimensions, e.g. a (N,) bias added to
    an (M, N) matrix in row-major layout). `small_is_left` flips operand order so
    non-commutative ops (sub, div) stay correct.
    """
    if cuda is None:
        raise RuntimeError("GPU not available: cannot launch kernels")
    assert big.on_gpu and small.on_gpu, "Both tensors must be on GPU"
    n = big.size
    m = small.size
    if small_is_left:
        body = f"out[idx] = b[idx % m] {op_symbol} a[idx];"
    else:
        body = f"out[idx] = a[idx] {op_symbol} b[idx % m];"
    kernel_src = f"""
    __global__ void {op_name}(const float* a, const float* b, float* out, int n, int m) {{
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {{ {body} }}
    }}
    """
    func = get_kernel(op_name, kernel_src)
    out_gpu = mempool.alloc(n * 4)
    bs = _optimal_block_size()
    block = (bs, 1, 1)
    grid = ((n + bs - 1) // bs, 1, 1)
    stream = _get_stream()
    func(big.gpu_ptr, small.gpu_ptr, out_gpu, np.int32(n), np.int32(m),
         block=block, grid=grid, stream=stream)

    Tensor = _tensor_cls()
    result = Tensor(out_gpu, gpu=True, inputs=[big, small])
    result.shape = big.shape
    result.size = n
    result.dtype = np.float32
    return result


def launch_reduce(a, op_name: str, reduce_type: str):
    """
    Two-pass parallel reduction using warp-shuffle instructions.

    Uses a grid-stride loop for the load phase (one kernel regardless of input
    size) and ``__shfl_down_sync`` for intra-warp communication, which avoids
    shared-memory bank conflicts and removes most ``__syncthreads`` calls vs the
    old tree-reduction approach. Always returns a float32 scalar Tensor.
    """
    if cuda is None:
        raise RuntimeError("GPU not available: cannot launch kernels")
    assert a.on_gpu, "Input tensor must be on GPU"

    if getattr(a, 'dtype', np.float32) == np.float16:
        a = _fp16_to_fp32(a)

    if reduce_type == "sum":
        init_val = "0.0f"
        accum = "val += in[i];"
        shfl_red = "val += __shfl_down_sync(0xffffffff, val, offset);"
    elif reduce_type == "max":
        init_val = "-3.402823e+38f"
        accum = "val = fmaxf(val, in[i]);"
        shfl_red = "val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));"
    elif reduce_type == "min":
        init_val = "3.402823e+38f"
        accum = "val = fminf(val, in[i]);"
        shfl_red = "val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));"
    else:
        raise ValueError(f"Unknown reduce_type: {reduce_type}")

    BS = 256
    N_WARPS = BS // 32   # 8 warps per block

    kernel_src = f"""
    __global__ void {op_name}(const float* in, float* out, int n) {{
        float val = {init_val};
        // Grid-stride accumulation — each thread covers multiple elements.
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {{
            {accum}
        }}
        // Warp-level reduction via shuffle (no __syncthreads inside a warp).
        for (int offset = 16; offset > 0; offset >>= 1) {{
            {shfl_red}
        }}
        // Collect one value per warp in shared memory.
        __shared__ float warp_sums[{N_WARPS}];
        int lane    = threadIdx.x & 31;
        int warp_id = threadIdx.x >> 5;
        if (lane == 0) warp_sums[warp_id] = val;
        __syncthreads();
        // Final reduction across warps — run entirely in warp 0.
        if (warp_id == 0) {{
            val = (lane < {N_WARPS}) ? warp_sums[lane] : {init_val};
            for (int offset = 16; offset > 0; offset >>= 1) {{
                {shfl_red}
            }}
            if (lane == 0) out[blockIdx.x] = val;
        }}
    }}
    """

    n = a.size
    grid_size = min((n + BS - 1) // BS, 1024)
    partial_gpu = mempool.alloc(grid_size * 4)
    func = get_kernel(op_name, kernel_src)
    stream = _get_stream()
    func(a.gpu_ptr, partial_gpu, np.int32(n),
         block=(BS, 1, 1), grid=(grid_size, 1, 1), stream=stream)

    if grid_size == 1:
        final_gpu = partial_gpu
    else:
        # Second pass: reduce at most 1024 block-partial values → 1 scalar.
        func2_name = op_name + "_p2"
        func2_src = kernel_src.replace(op_name, func2_name)
        func2 = get_kernel(func2_name, func2_src)
        final_gpu = mempool.alloc(4)
        func2(partial_gpu, final_gpu, np.int32(grid_size),
              block=(BS, 1, 1), grid=(1, 1, 1), stream=stream)

    Tensor = _tensor_cls()
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
    Tensor = _tensor_cls()
    result = Tensor(out_gpu, gpu=True, inputs=[a])
    result.shape = a.shape
    result.size = n
    result.dtype = np.float32
    return result


# ---------------------------------------------------------------------------
# Item 0: cuBLAS-backed matmul (skcuda or ctypes fallback, with TF32 on Ampere+)
# ---------------------------------------------------------------------------

_cublas_handle = None    # persistent handle — skcuda int or ctypes c_void_p value
_cublas_backend = None   # 'skcuda' | 'ctypes'
_ctypes_cublas_lib = None


def _get_ctypes_cublas():
    """Load libcublas via ctypes. Returns the CDLL or None."""
    global _ctypes_cublas_lib
    if _ctypes_cublas_lib is not None:
        return _ctypes_cublas_lib
    import ctypes
    for name in ('libcublas.so.12', 'libcublas.so.11', 'libcublas.so.10', 'libcublas.so'):
        try:
            _ctypes_cublas_lib = ctypes.CDLL(name)
            return _ctypes_cublas_lib
        except Exception:
            pass
    return None


def _enable_tf32(handle_val, lib_or_skcuda):
    """Best-effort: enable TF32 math mode (CUBLAS_TF32_TENSOR_OP_MATH = 3)."""
    # TF32 is only available on Ampere+ (sm_80+); no-op on older GPUs.
    try:
        if _cublas_backend == 'skcuda':
            lib_or_skcuda.cublasSetMathMode(handle_val, 3)
        else:
            import ctypes
            lib_or_skcuda.cublasSetMathMode(
                ctypes.c_void_p(handle_val), ctypes.c_int(3))
    except Exception:
        pass


def _get_cublas_handle():
    """Lazily create and cache a single cuBLAS handle. None if unavailable."""
    global _cublas_handle, _cublas_backend
    if _cublas_handle is not None:
        return _cublas_handle

    # --- path 1: skcuda ---
    try:
        from skcuda import cublas as sk_cublas
        h = sk_cublas.cublasCreate()
        _cublas_handle = h
        _cublas_backend = 'skcuda'
        _enable_tf32(h, sk_cublas)
        import atexit
        atexit.register(_destroy_cublas_handle)
        return _cublas_handle
    except Exception:
        pass

    # --- path 2: ctypes libcublas ---
    lib = _get_ctypes_cublas()
    if lib is not None:
        import ctypes
        handle_ptr = ctypes.c_void_p()
        try:
            if lib.cublasCreate_v2(ctypes.byref(handle_ptr)) == 0 and handle_ptr.value:
                _cublas_handle = handle_ptr.value
                _cublas_backend = 'ctypes'
                _enable_tf32(_cublas_handle, lib)
                import atexit
                atexit.register(_destroy_cublas_handle)
                return _cublas_handle
        except Exception:
            pass

    return None


def _destroy_cublas_handle():
    global _cublas_handle, _cublas_backend
    if _cublas_handle is None:
        return
    try:
        if _cublas_backend == 'skcuda':
            from skcuda import cublas as sk_cublas
            sk_cublas.cublasDestroy(_cublas_handle)
        else:
            import ctypes
            lib = _get_ctypes_cublas()
            if lib is not None:
                lib.cublasDestroy_v2(ctypes.c_void_p(_cublas_handle))
    except Exception:
        pass
    finally:
        _cublas_handle = None
        _cublas_backend = None


def _launch_matmul_cublas(a, b):
    """
    Matrix multiplication via cuBLAS cublasSgemm.
    Tries skcuda first, then a ctypes-based fallback (no extra deps required).
    Returns None if cuBLAS is unavailable in this environment.
    """
    handle = _get_cublas_handle()
    if handle is None:
        return None

    M, K = a.shape
    _, N = b.shape
    stream = _get_stream()

    if _cublas_backend == 'skcuda':
        from skcuda import cublas as sk_cublas
        try:
            if stream is not None:
                try:
                    sk_cublas.cublasSetStream(handle, stream.handle)
                except Exception:
                    pass
            out_gpu = mempool.alloc(M * N * 4)
            # cuBLAS is column-major: C = A@B (row-major) ↔ C^T = B^T @ A^T (col-major)
            sk_cublas.cublasSgemm(
                handle,
                'n', 'n',
                N, M, K,
                np.float32(1.0),
                b.gpu_ptr, N,
                a.gpu_ptr, K,
                np.float32(0.0),
                out_gpu, N,
            )
            Tensor = _tensor_cls()
            result = Tensor(out_gpu, gpu=True, inputs=[a, b])
            result.shape = (M, N)
            result.size = M * N
            result.dtype = np.float32
            return result
        except Exception:
            return None

    # ctypes cuBLAS path (no skcuda dependency required)
    import ctypes
    lib = _ctypes_cublas_lib
    try:
        if stream is not None:
            try:
                lib.cublasSetStream_v2(
                    ctypes.c_void_p(handle),
                    ctypes.c_void_p(stream.handle))
            except Exception:
                pass
        out_gpu = mempool.alloc(M * N * 4)
        alpha = ctypes.c_float(1.0)
        beta  = ctypes.c_float(0.0)
        # Column-major trick: C = A@B ↔ C^T = B^T @ A^T
        ret = lib.cublasSgemm_v2(
            ctypes.c_void_p(handle),
            ctypes.c_int(0), ctypes.c_int(0),   # CUBLAS_OP_N, CUBLAS_OP_N
            ctypes.c_int(N), ctypes.c_int(M), ctypes.c_int(K),
            ctypes.byref(alpha),
            ctypes.c_void_p(int(b.gpu_ptr)), ctypes.c_int(N),
            ctypes.c_void_p(int(a.gpu_ptr)), ctypes.c_int(K),
            ctypes.byref(beta),
            ctypes.c_void_p(int(out_gpu)), ctypes.c_int(N),
        )
        if ret != 0:
            return None
        Tensor = _tensor_cls()
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

    Tensor = _tensor_cls()
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
    Matmul + bias: x(M,K) @ w(K,N) + bias(N,) → (M,N).

    Uses cuBLAS GEMM (fast, TF32 on Ampere+) when available, followed by a
    broadcast-add kernel for the bias. Falls back to a single tiled kernel
    when cuBLAS is unavailable.
    """
    if cuda is None:
        raise RuntimeError("GPU not available: cannot launch kernels")
    assert x.on_gpu and w.on_gpu and bias.on_gpu, "All tensors must be on GPU"
    M, K = x.shape
    K2, N = w.shape
    assert K == K2, f"Shape mismatch: {x.shape} @ {w.shape}"
    assert bias.size == N, f"Bias size {bias.size} must match output columns {N}"

    # Fast path: cuBLAS GEMM + fused broadcast-add kernel for the bias.
    mm = _launch_matmul_cublas(x, w)
    if mm is not None:
        return launch_fused([mm, bias], f"x0[idx] + x1[idx % {N}]", "mb_fused")

    # Tiled fallback: GEMM and bias fused in one kernel.
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

    Tensor = _tensor_cls()
    result = Tensor(out_gpu, gpu=True, inputs=[x, w, bias])
    result.shape = (M, N)
    result.size = M * N
    result.dtype = np.float32
    return result


def launch_matmul_bias_relu(x, w, bias):
    """
    Matmul + bias + ReLU: x(M,K) @ w(K,N) + bias(N,) → relu → (M,N).

    Uses cuBLAS GEMM + one fused bias+relu broadcast kernel when cuBLAS is
    available. Falls back to a single tiled kernel otherwise.
    """
    if cuda is None:
        raise RuntimeError("GPU not available: cannot launch kernels")
    assert x.on_gpu and w.on_gpu and bias.on_gpu, "All tensors must be on GPU"
    M, K = x.shape
    K2, N = w.shape
    assert K == K2, f"Shape mismatch: {x.shape} @ {w.shape}"
    assert bias.size == N, f"Bias size {bias.size} must match output columns {N}"

    # Fast path: cuBLAS GEMM + fused bias+relu broadcast kernel.
    mm = _launch_matmul_cublas(x, w)
    if mm is not None:
        return launch_fused(
            [mm, bias],
            f"fmaxf(0.0f, x0[idx] + x1[idx % {N}])",
            "mbr_fused")

    # Tiled fallback: GEMM, bias, and relu fused in one kernel.
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

    Tensor = _tensor_cls()
    result = Tensor(out_gpu, gpu=True, inputs=[x, w, bias])
    result.shape = (M, N)
    result.size = M * N
    result.dtype = np.float32
    return result
