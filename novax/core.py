import numpy as np
from novax.utils import mempool

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import nvrtc
    GPU_AVAILABLE = True 
except ImportError:
    GPU_AVAILABLE = False

class Tensor:
    """
    NovaX tensor class: CPU-GPU backed tensor class
    """
    def __init__(self, data, op=None, inputs=None):
        # operation graph info
        self.op = op
        self.inputs = inputs or []
        self.is_leaf = op is None

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

        # device info
        self.gpu_ptr = None
        self.on_gpu = False

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
        if not GPU_AVAILABLE:
            return self.data
        out = np.empty_like(self.data)
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
            if self.op == "add":
                return Tensor(left.data + right.data)
            elif self.op == "mul":
                return Tensor(left.data * right.data)
            elif self.op == "sub":
                return Tensor(left.data - right.data)
            elif self.op == "div":
                return Tensor(left.data / right.data) 

        # JIT compile a fused CUDA kernel for the operation
        kernel_src = f"""
        extern "C" __global__
        void fused_kernel(const float* a, const float* b, float* out, int n) {{
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < n) {{
                out[idx] = {self._emit_expr()};
            }}
        }}
        """
        prog = nvrtc.Program(kernel_src, "fused.cu", [])
        ptx = prog.compile([])
        mod = cuda.module_from_buffer(ptx.encode("utf-8"))
        func = mod.get_function("fused_kernel")

        out = np.empty_like(left.data)
        out_gpu = cuda.mem_alloc(out.nbytes) 

        block = (256, 1, 1)
        grid = ((left.size + block[0] - 1) // block[0], 1, 1)

        func(left.gpu_ptr, right.gpu_ptr, out_gpu, np.int32(left.size), 
             block=(block,1,1), grid=(grid,1,1), stream=None)

        cuda.memcpy_dtoh(out, out_gpu)
        return Tensor(out)

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

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self.op}, leaf={self.is_leaf})"

    # ---------------------------
    # Context Manager Support
    # ---------------------------

    def free(self):
        if self.gpu_ptr is not None:
            nbytes = self.size * 4
            mempool.free(self.gpu_ptr, nbytes)
            self.gpu_ptr = None
            self.on_gpu = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.free()
        except Exception:
            pass


