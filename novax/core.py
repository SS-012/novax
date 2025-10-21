import numpy as np
from novax.utils import mempool
from novax.ops.launcher import launch_kernel, launch_fused

try:
    import pycuda.driver as cuda
    cuda.init()
    if not cuda.Context.get_current():
        ctx = cuda.Device(0).make_context()
    GPU_AVAILABLE = True 
except ImportError:
    cuda = None
    GPU_AVAILABLE = False

try:
    from cuda import nvrtc
except ImportError:
    nvrtc = None


class Tensor:
    """
    NovaX tensor class: CPU-GPU backed tensor class
    """
    def __init__(self, data, op=None, inputs=None, gpu=False):
        # detect constants
        self.is_constant = False
        self.const_value = None
        if isinstance(data, (int, float, np.generic)):
            self.is_constant = True
            self.const_value = float(data)
            self.shape = (1,) 
            self.size = 1
            self.dtype = np.float32
            self.data = np.array([data], dtype=np.float32)
            self.gpu_ptr = None
            self.on_gpu = False
            self.is_leaf = True
            return  # short-circuit here; don’t process host array creation


        # operation graph info
        self.op = op
        self.inputs = inputs or []
        self.is_leaf = op is None

        # device info
        self.gpu_ptr = None
        self.on_gpu = gpu

        if gpu:
            # Case 1: this tensor wraps an existing GPU allocation
            self.data = None  # no host data stored
            if self.inputs:
                # infer shape from parent tensor(s)
                self.shape = self.inputs[0].shape
                self.dtype = self.inputs[0].dtype
                self.size = self.inputs[0].size
            else:
                # shape must be provided externally if no inputs
                self.shape = getattr(data, "shape", None)
                self.dtype = np.float32
                self.size = np.prod(self.shape) if self.shape is not None else None
            self.gpu_ptr = data  # data is actually a device pointer
            return  # short-circuit here; don’t process host array creation

        # data & metadata
        if data is None:
            self.data = None
            if self.inputs:
                # Infer from first input (assumes elementwise shape agreement)
                self.shape = self.inputs[0].shape
                self.dtype = self.inputs[0].dtype
                self.size = self.inputs[0].size
            else:
                self.shape = None
                self.dtype = None
                self.size = None
        else:
            if isinstance(data, (list, tuple)):
                self.data = np.array(data, dtype=np.float32)
            elif isinstance(data, np.ndarray):
                self.data = data.astype(np.float32)
            else:
                self.data = np.array(data, dtype=np.float32)

            # tensor metadata
            self.shape = self.data.shape
            self.dtype = self.data.dtype
            self.size = self.data.size

 

    # ---------------------------
    #  GPU transfer operations
    # ---------------------------
    def _to_gpu(self):
        self.nbytes = self.data.nbytes
        self.gpu_ptr = cuda.mem_alloc(self.data.nbytes)
        cuda.memcpy_htod(self.gpu_ptr, self.data)
        self.on_gpu = True

    def to_gpu(self):
        """
        Explicitly upload tensor data to GPU. Only valid when data is present.
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU not available")
        if self.data is None:
            raise ValueError("Cannot upload a tensor without host data")
        if self.gpu_ptr is not None:
            return self
        self._to_gpu()
        return self

    def to_host(self):
        if not self.on_gpu or not GPU_AVAILABLE:
            return self.data
        if self.gpu_ptr is None:
            raise RuntimeError("Cannot call to_host() - tensor not on GPU.")
        # Ensure our stream finished
        try:
            from novax.ops.launcher import _get_stream
            s = _get_stream()
            if s is not None:
                s.synchronize()
        except Exception:
            cuda.Context.synchronize()
        out = np.empty(self.shape, self.dtype)
        cuda.memcpy_dtoh(out, self.gpu_ptr)
        return out


    # ---------------------------
    # Lazy Operations
    # ---------------------------
    def __add__(self, other):
        return Tensor(None, op="add", inputs=[self, self._wrap(other)])

    def __mul__(self, other):
        return Tensor(None, op="mul", inputs=[self, self._wrap(other)])

    def __sub__(self, other):
        return Tensor(None, op="sub", inputs=[self, self._wrap(other)])

    def __truediv__(self, other):
        return Tensor(None, op="div", inputs=[self, self._wrap(other)])

    def _wrap(self, val):
        return val if isinstance(val, Tensor) else Tensor(val)

    # ---------------------------
    # Evaluation / JIT Compilation
    # ---------------------------
    def eval(self):
        """
        Compile and execute the entire expression graph
        """
        if self.is_leaf:
            return self 

        # recursively evaluate all inputs
        left = self.inputs[0].eval()
        right = self.inputs[1].eval()

        if (not GPU_AVAILABLE) or (left.gpu_ptr is None) or (right.gpu_ptr is None):
            # CPU evaluation path (also used when inputs are not explicitly on GPU)
            left_arr = left.to_host() if left.on_gpu else left.data
            right_arr = right.to_host() if right.on_gpu else right.data
            if self.op == "add":
                return Tensor(left_arr + right_arr)
            elif self.op == "mul":
                return Tensor(left_arr * right_arr)
            elif self.op == "sub":
                return Tensor(left_arr - right_arr)
            elif self.op == "div":
                return Tensor(left_arr / right_arr) 

        # GPU fused path: attempt to fuse entire subgraph when all leaves are GPU
        folded = self._fold_constants()
        fused_expr, leaves = folded._build_fused()
        if fused_expr is not None and all(t.on_gpu for t in leaves):
            return launch_fused(leaves, fused_expr, "fused_kernel")

        # Fallback: binary fused kernel or scalar op
        if self.op in ("mul", "div") and getattr(self.inputs[1], "is_constant", False):
            scalar = float(self.inputs[1].const_value)
            expr = "a[idx] * s" if self.op == "mul" else "a[idx] / s"
            return launch_kernel(left, None, "fused_kernel_scalar", expr, scalar=scalar)

    def _emit_expr(self):
        """
        Emit the CUDA kernel code for the operation
        """
        op_map = {
            "add": "a[idx] + b[idx]",
            "mul": "a[idx] * b[idx]",
            "sub": "a[idx] - b[idx]",
            "div": "a[idx] / b[idx]",
        }
        return op_map[self.op]

    def _build_fused(self):
        leaves = []
        index_map = {}

        def register_leaf(t: "Tensor") -> str:
            if t.is_constant:
                # inline the scalar constant
                return f"{t.const_value:.8f}f"
            if t not in index_map:
                index_map[t] = len(leaves)
                leaves.append(t)
            return f"x{index_map[t]}[idx]"

        def build(node: "Tensor") -> str | None:
            if node.is_leaf:
                return register_leaf(node)
            if node.op not in ("add", "sub", "mul", "div"):
                return None
            left_expr = build(node.inputs[0])
            right_expr = build(node.inputs[1])
            if left_expr is None or right_expr is None:
                return None
            op_map = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
            return f"({left_expr} {op_map[node.op]} {right_expr})"

        expr = build(self)
        if expr is None:
            return None, []
        return expr, leaves

    def _fold_constants(self):
        if self.is_leaf:
            return self
        left = self.inputs[0]._fold_constants()
        right = self.inputs[1]._fold_constants()

        if left.is_constant and right.is_constant:
            val = {
                "add": left.const_value + right.const_value,
                "sub": left.const_value - right.const_value,
                "mul": left.const_value * right.const_value,
                "div": left.const_value / right.const_value,
            }[self.op]
            return Tensor(val)
        return Tensor(None, op=self.op, inputs=[left, right])


    def __repr__(self):
        device = "GPU" if self.on_gpu else "CPU"
        shape = self.shape if self.shape is not None else "?"
        return f"Tensor(shape={shape}, device={device}, op={self.op}, leaf={self.is_leaf})"

    # ---------------------------
    # Context Manager Support
    # ---------------------------

    def free(self, release=False):
        """
        Frees the GPU buffer for this tensor.
        By default, caches the memory for reuse.
        Set release=True to actually free it to the CUDA driver.
        """
        if self.on_gpu and self.gpu_ptr is not None:
            try:
                mempool.free(self.gpu_ptr, self.size * 4, release=release)
            except Exception:
                pass
            finally:
                self.gpu_ptr = None
                self.on_gpu = False


    def release_context(self):
        if cuda and cuda.Context.get_current():
            cuda.Context.pop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.free()
        except Exception:
            pass


