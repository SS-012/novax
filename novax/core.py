import atexit

import numpy as np
from novax.utils import mempool
from novax.ops.launcher import launch_kernel, launch_fused

_CUDA_CONTEXT = None

try:
    import pycuda.driver as cuda
    cuda.init()
    if not cuda.Context.get_current():
        _CUDA_CONTEXT = cuda.Device(0).retain_primary_context()
        _CUDA_CONTEXT.push()

        def _cleanup_cuda_context():
            global _CUDA_CONTEXT
            if _CUDA_CONTEXT is not None:
                try:
                    cuda.Context.pop()
                except Exception:
                    pass
                try:
                    _CUDA_CONTEXT.detach()
                except Exception:
                    pass
                _CUDA_CONTEXT = None

        atexit.register(_cleanup_cuda_context)
    GPU_AVAILABLE = True
except ImportError:
    cuda = None
    GPU_AVAILABLE = False

try:
    from cuda import nvrtc
except ImportError:
    nvrtc = None

_UNARY_ELEMENTWISE = frozenset({"exp", "log", "sqrt", "abs", "neg", "relu", "sigmoid", "tanh", "softmax"})
_REDUCE_OPS = frozenset({"sum", "mean", "max", "min"})
_BINARY_OPS = frozenset({"add", "sub", "mul", "div", "pow"})

_UNARY_CUDA_EXPR = {
    "exp":     "expf(a[idx])",
    "log":     "logf(a[idx])",
    "sqrt":    "sqrtf(a[idx])",
    "abs":     "fabsf(a[idx])",
    "neg":     "(-a[idx])",
    "relu":    "fmaxf(0.0f, a[idx])",
    "sigmoid": "(1.0f / (1.0f + expf(-a[idx])))",
    "tanh":    "tanhf(a[idx])",
}

_UNARY_NUMPY = {
    "exp":     lambda x: np.exp(x),
    "log":     lambda x: np.log(x),
    "sqrt":    lambda x: np.sqrt(x),
    "abs":     lambda x: np.abs(x),
    "neg":     lambda x: -x,
    "relu":    lambda x: np.maximum(0.0, x),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "tanh":    lambda x: np.tanh(x),
    "softmax": lambda x: (lambda e: e / np.sum(e))(np.exp(x - np.max(x))),
    "sum":     lambda x: np.array([np.sum(x)], dtype=np.float32),
    "mean":    lambda x: np.array([np.mean(x)], dtype=np.float32),
    "max":     lambda x: np.array([np.max(x)], dtype=np.float32),
    "min":     lambda x: np.array([np.min(x)], dtype=np.float32),
}


class Tensor:
    """NovaX tensor: CPU/GPU backed, lazy evaluation, autograd support."""

    def __init__(self, data, op=None, inputs=None, gpu=False, requires_grad=False):
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev: set = set()

        # --- scalar / constant ---
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
            self.op = op
            self.inputs = inputs or []
            return

        # --- operation graph info ---
        self.op = op
        self.inputs = inputs or []
        self.is_leaf = op is None

        # --- device state ---
        self.gpu_ptr = None
        self.on_gpu = gpu

        if gpu:
            self.data = None
            if self.inputs:
                self.shape = self.inputs[0].shape
                self.dtype = self.inputs[0].dtype
                self.size = self.inputs[0].size
            else:
                self.shape = getattr(data, "shape", None)
                self.dtype = np.float32
                self.size = int(np.prod(self.shape)) if self.shape is not None else None
            self.gpu_ptr = data
            return

        # --- host data ---
        if data is None:
            self.data = None
            if self.inputs:
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
            self.shape = self.data.shape
            self.dtype = self.data.dtype
            self.size = self.data.size

    # ------------------------------------------------------------------
    # GPU transfer
    # ------------------------------------------------------------------

    def _to_gpu(self):
        self.gpu_ptr = mempool.alloc(self.data.nbytes)
        cuda.memcpy_htod(self.gpu_ptr, self.data)
        self.data = None
        self.on_gpu = True

    def to_gpu(self):
        """Upload tensor data to GPU memory."""
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU not available")
        if self.gpu_ptr is not None:
            return self
        if self.data is None:
            raise ValueError("Cannot upload a tensor without host data")
        self._to_gpu()
        return self

    def to_host(self):
        """Download tensor data to a numpy array."""
        if not self.is_leaf and self.data is None and self.gpu_ptr is None:
            return self.eval().to_host()
        if not self.on_gpu or not GPU_AVAILABLE:
            return self.data
        if self.gpu_ptr is None:
            raise RuntimeError("Cannot call to_host() — tensor not on GPU.")
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

    # ------------------------------------------------------------------
    # Lazy operator overloading
    # ------------------------------------------------------------------

    def __add__(self, other):
        return Tensor(None, op="add", inputs=[self, self._wrap(other)])

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        return Tensor(None, op="mul", inputs=[self, self._wrap(other)])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return Tensor(None, op="sub", inputs=[self, self._wrap(other)])

    def __rsub__(self, other):
        return Tensor(other).__sub__(self)

    def __truediv__(self, other):
        return Tensor(None, op="div", inputs=[self, self._wrap(other)])

    def __rtruediv__(self, other):
        return Tensor(other).__truediv__(self)

    def __neg__(self):
        return Tensor(None, op="neg", inputs=[self])

    def __matmul__(self, other):
        return Tensor(None, op="matmul", inputs=[self, self._wrap(other)])

    def __pow__(self, other):
        return Tensor(None, op="pow", inputs=[self, self._wrap(other)])

    def _wrap(self, val):
        return val if isinstance(val, Tensor) else Tensor(val)

    # ------------------------------------------------------------------
    # Shape manipulation
    # ------------------------------------------------------------------

    def reshape(self, *new_shape):
        """Return a new Tensor with data reshaped. Downloads from GPU if needed."""
        arr = self.to_host() if self.on_gpu else self.data
        if arr is None:
            raise ValueError("Cannot reshape a tensor without host data")
        return Tensor(arr.reshape(new_shape))

    def transpose(self, axes=None):
        """Return a transposed Tensor. Downloads from GPU if needed."""
        arr = self.to_host() if self.on_gpu else self.data
        if arr is None:
            raise ValueError("Cannot transpose a tensor without host data")
        return Tensor(np.transpose(arr, axes))

    # ------------------------------------------------------------------
    # Evaluation / JIT compilation
    # ------------------------------------------------------------------

    def eval(self):
        """Compile and execute the expression graph, returning a concrete Tensor."""
        from novax.autograd import _get_grad_enabled

        if self.is_leaf:
            return self

        track_grad = _get_grad_enabled()
        if self.op in _UNARY_ELEMENTWISE:
            fused = self._try_eval_fused_elementwise(track_grad)
            if fused is not None:
                return fused

            inp = self.inputs[0].eval()
            return self._eval_unary(inp, track_grad)

        if self.op in _REDUCE_OPS:
            inp = self.inputs[0].eval()
            return self._eval_unary(inp, track_grad)

        if self.op == "matmul":
            left = self.inputs[0].eval()
            right = self.inputs[1].eval()
            return self._eval_matmul(left, right, track_grad)

        # Binary elementwise: add, sub, mul, div, pow
        fused = self._try_eval_fused_elementwise(track_grad)
        if fused is not None:
            return fused

        left = self.inputs[0].eval()
        right = self.inputs[1].eval()
        return self._eval_binary(left, right, track_grad)

    def _try_eval_fused_elementwise(self, track_grad: bool):
        if (not GPU_AVAILABLE) or self.op not in (_UNARY_ELEMENTWISE | _BINARY_OPS):
            return None

        specialized = self._try_eval_special_fused_elementwise(track_grad)
        if specialized is not None:
            return specialized

        folded = self._fold_constants()
        fused_expr, leaves = folded._build_fused()
        if fused_expr is None or not leaves:
            return None
        if not all(getattr(t, "on_gpu", False) for t in leaves):
            return None
        if track_grad and any(getattr(t, "requires_grad", False) for t in leaves):
            return None

        try:
            return launch_fused(leaves, fused_expr, "fused_kernel")
        except Exception:
            return None

    def _try_eval_special_fused_elementwise(self, track_grad: bool):
        if track_grad:
            return None

        def same_shape_gpu(*tensors):
            return (
                tensors
                and all(getattr(t, "is_leaf", False) for t in tensors)
                and all(getattr(t, "on_gpu", False) for t in tensors)
                and all(not getattr(t, "requires_grad", False) for t in tensors)
                and all(t.shape == tensors[0].shape for t in tensors)
            )

        def match_mul_add_relu(node):
            if getattr(node, "op", None) != "relu":
                return None
            add = node.inputs[0]
            if getattr(add, "op", None) != "add":
                return None
            left, right = add.inputs
            if getattr(left, "op", None) == "mul":
                mul, c = left, right
            elif getattr(right, "op", None) == "mul":
                mul, c = right, left
            else:
                return None
            a, b = mul.inputs
            return (a, b, c) if same_shape_gpu(a, b, c) else None

        try:
            relu_args = match_mul_add_relu(self)
            if relu_args is not None:
                from novax.ops.launcher import launch_relu_mul_add
                return launch_relu_mul_add(*relu_args)

            if self.op != "sigmoid":
                return None
            mul = self.inputs[0]
            if getattr(mul, "op", None) != "mul":
                return None
            left, right = mul.inputs
            if getattr(left, "op", None) == "relu":
                relu, tail = left, right
            elif getattr(right, "op", None) == "relu":
                relu, tail = right, left
            else:
                return None
            relu_args = match_mul_add_relu(relu)
            if relu_args is None:
                return None
            a, b, c = relu_args
            if tail is not a:
                return None
            from novax.ops.launcher import launch_sigmoid_relu_mul_add_mul
            return launch_sigmoid_relu_mul_add_mul(a, b, c)
        except Exception:
            return None

    def _eval_unary(self, inp, track_grad: bool):
        op = self.op
        # GPU path
        if GPU_AVAILABLE and inp.on_gpu and op in _UNARY_ELEMENTWISE:
            try:
                out = launch_kernel(inp, None, f"{op}_kernel", _UNARY_CUDA_EXPR[op])
                self._attach_unary_grad(out, inp, op, track_grad)
                return out
            except Exception:
                pass
        if GPU_AVAILABLE and inp.on_gpu and op in _REDUCE_OPS:
            try:
                from novax.ops.launcher import launch_reduce
                if op == "sum":
                    out = launch_reduce(inp, "sum_reduce", "sum")
                elif op == "mean":
                    out = launch_reduce(
                        inp,
                        "mean_sum_reduce",
                        "sum",
                        scale=1.0 / float(inp.size),
                    )
                elif op == "max":
                    out = launch_reduce(inp, "max_reduce", "max")
                elif op == "min":
                    out = launch_reduce(inp, "min_reduce", "min")
                self._attach_unary_grad(out, inp, op, track_grad)
                return out
            except Exception:
                pass
        # CPU fallback
        arr = inp.to_host() if inp.on_gpu else inp.data
        result = _UNARY_NUMPY[op](arr)
        if not isinstance(result, np.ndarray):
            result = np.array([result], dtype=np.float32)
        out = Tensor(result.astype(np.float32))
        self._attach_unary_grad(out, inp, op, track_grad)
        return out

    def _eval_matmul(self, left, right, track_grad: bool):
        if GPU_AVAILABLE and left.on_gpu and right.on_gpu:
            try:
                from novax.ops.launcher import launch_matmul
                out = launch_matmul(left, right)
                self._attach_matmul_grad(out, left, right, track_grad)
                return out
            except Exception:
                pass
        left_arr = left.to_host() if left.on_gpu else left.data
        right_arr = right.to_host() if right.on_gpu else right.data
        out = Tensor(np.matmul(left_arr, right_arr))
        self._attach_matmul_grad(out, left, right, track_grad)
        return out

    def _eval_binary(self, left, right, track_grad: bool):
        op = self.op
        # CPU path
        if (not GPU_AVAILABLE) or (left.gpu_ptr is None) or (right.gpu_ptr is None):
            left_arr = left.to_host() if left.on_gpu else left.data
            right_arr = right.to_host() if right.on_gpu else right.data
            _cpu_binary = {
                "add": left_arr + right_arr,
                "sub": left_arr - right_arr,
                "mul": left_arr * right_arr,
                "div": left_arr / right_arr,
                "pow": np.power(left_arr, right_arr),
            }
            out = Tensor(_cpu_binary[op])
            self._attach_binary_grad(out, left, right, op, track_grad)
            return out

        # GPU fused path
        folded = self._fold_constants()
        fused_expr, leaves = folded._build_fused()
        if fused_expr is not None and all(t.on_gpu for t in leaves):
            out = launch_fused(leaves, fused_expr, "fused_kernel")
            self._attach_binary_grad(out, left, right, op, track_grad)
            return out

        # Scalar shortcut for mul/div with constant RHS
        if op in ("mul", "div") and getattr(self.inputs[1], "is_constant", False):
            scalar = float(self.inputs[1].const_value)
            expr = "a[idx] * s" if op == "mul" else "a[idx] / s"
            out = launch_kernel(left, None, "fused_kernel_scalar", expr, scalar=scalar)
            self._attach_binary_grad(out, left, right, op, track_grad)
            return out

        # General binary GPU fallback
        op_symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": None}[op]
        if op == "pow":
            out = launch_kernel(left, right, "pow_kernel", "powf(a[idx], b[idx])")
        else:
            out = launch_kernel(left, right, f"{op}_kernel", f"a[idx] {op_symbol} b[idx]")
        self._attach_binary_grad(out, left, right, op, track_grad)
        return out

    # ------------------------------------------------------------------
    # Autograd backward setup helpers
    # ------------------------------------------------------------------

    def _attach_unary_grad(self, out, inp, op, track_grad):
        if track_grad:
            from novax.autograd import attach_unary_grad
            attach_unary_grad(out, inp, op)

    def _attach_matmul_grad(self, out, left, right, track_grad):
        if track_grad:
            from novax.autograd import attach_matmul_grad
            attach_matmul_grad(out, left, right)

    def _attach_binary_grad(self, out, left, right, op, track_grad):
        if track_grad:
            from novax.autograd import attach_binary_grad
            attach_binary_grad(out, left, right, op)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self):
        """Reverse-mode autodiff. Seeds self.grad = ones, propagates backward."""
        topo = []
        visited = set()

        def _build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for prev in v._prev:
                    _build_topo(prev)
                topo.append(v)

        _build_topo(self)
        self.grad = Tensor(np.ones(self.shape, dtype=np.float32))
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------
    # Expression graph helpers
    # ------------------------------------------------------------------

    def _emit_expr(self):
        op_map = {
            "add": "a[idx] + b[idx]",
            "mul": "a[idx] * b[idx]",
            "sub": "a[idx] - b[idx]",
            "div": "a[idx] / b[idx]",
            "pow": "powf(a[idx], b[idx])",
        }
        return op_map.get(self.op)

    def _build_fused(self):
        leaves = []
        index_map = {}

        def register_leaf(t: "Tensor") -> str:
            if t.is_constant:
                return f"{t.const_value:.8f}f"
            if t not in index_map:
                index_map[t] = len(leaves)
                leaves.append(t)
            return f"x{index_map[t]}[idx]"

        _binary_sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
        _unary_cuda = {
            "exp":     "expf({e})",
            "log":     "logf({e})",
            "sqrt":    "sqrtf({e})",
            "abs":     "fabsf({e})",
            "neg":     "(-({e}))",
            "relu":    "fmaxf(0.0f, {e})",
            "sigmoid": "(1.0f / (1.0f + expf(-({e}))))",
            "tanh":    "tanhf({e})",
        }

        def build(node: "Tensor"):
            if node.is_leaf:
                return register_leaf(node)
            if node.op in _binary_sym:
                le = build(node.inputs[0])
                re = build(node.inputs[1])
                if le is None or re is None:
                    return None
                return f"({le} {_binary_sym[node.op]} {re})"
            if node.op in _unary_cuda:
                sub = build(node.inputs[0])
                if sub is None:
                    return None
                return _unary_cuda[node.op].format(e=sub)
            return None

        expr = build(self)
        if expr is None:
            return None, []
        return expr, leaves

    def _fold_constants(self):
        if self.is_leaf:
            return self
        if self.op in _UNARY_ELEMENTWISE or self.op in _REDUCE_OPS:
            inp = self.inputs[0]._fold_constants()
            if inp.is_constant and self.op in _UNARY_NUMPY:
                val = float(_UNARY_NUMPY[self.op](np.array([inp.const_value], dtype=np.float32))[0])
                return Tensor(val)
            return Tensor(None, op=self.op, inputs=[inp])
        left = self.inputs[0]._fold_constants()
        right = self.inputs[1]._fold_constants()
        if left.is_constant and right.is_constant:
            ops = {
                "add": left.const_value + right.const_value,
                "sub": left.const_value - right.const_value,
                "mul": left.const_value * right.const_value,
                "div": left.const_value / right.const_value,
                "pow": left.const_value ** right.const_value,
            }
            if self.op in ops:
                return Tensor(ops[self.op])
        return Tensor(None, op=self.op, inputs=[left, right])

    # ------------------------------------------------------------------
    # Memory management & context manager
    # ------------------------------------------------------------------

    def free(self, release=False):
        """Return GPU buffer to pool. Set release=True to free to CUDA driver."""
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

    def __repr__(self):
        device = "GPU" if self.on_gpu else "CPU"
        shape = self.shape if self.shape is not None else "?"
        grad_info = f", grad={self.grad is not None}" if self.requires_grad else ""
        return f"Tensor(shape={shape}, device={device}, op={self.op}, leaf={self.is_leaf}{grad_info})"
