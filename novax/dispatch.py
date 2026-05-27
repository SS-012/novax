"""
Dispatch layer: routes NovaX operations to the appropriate backend.
Each function handles lazy tensor inputs (returning a lazy node) and sets up
autograd backward closures for concrete tensors that require gradients.
"""

try:
    from novax.core import GPU_AVAILABLE
except Exception:
    GPU_AVAILABLE = False

DEFAULT_DEVICE = "gpu" if GPU_AVAILABLE else "cpu"

try:
    from novax.ops.gpu import (
        add as gpu_add, sub as gpu_sub, mul as gpu_mul, div as gpu_div,
        exp as gpu_exp, log as gpu_log, sqrt as gpu_sqrt, abs as gpu_abs,
        neg as gpu_neg, pow as gpu_pow, relu as gpu_relu, sigmoid as gpu_sigmoid,
        tanh as gpu_tanh, softmax as gpu_softmax,
        sum as gpu_sum, mean as gpu_mean, max as gpu_max, min as gpu_min,
        matmul as gpu_matmul,
    )
except Exception:
    gpu_add = gpu_sub = gpu_mul = gpu_div = None
    gpu_exp = gpu_log = gpu_sqrt = gpu_abs = gpu_neg = gpu_pow = None
    gpu_relu = gpu_sigmoid = gpu_tanh = gpu_softmax = None
    gpu_sum = gpu_mean = gpu_max = gpu_min = None
    gpu_matmul = None

from novax.ops.cpu import (
    add as cpu_add, sub as cpu_sub, mul as cpu_mul, div as cpu_div,
    exp as cpu_exp, log as cpu_log, sqrt as cpu_sqrt, abs as cpu_abs,
    neg as cpu_neg, pow as cpu_pow, relu as cpu_relu, sigmoid as cpu_sigmoid,
    tanh as cpu_tanh, softmax as cpu_softmax,
    sum as cpu_sum, mean as cpu_mean, max as cpu_max, min as cpu_min,
    matmul as cpu_matmul,
)


def _use_gpu(a, b=None):
    if DEFAULT_DEVICE != "gpu" or not GPU_AVAILABLE:
        return False
    if not a.on_gpu:
        return False
    if b is not None and not b.on_gpu:
        return False
    return True


def _is_lazy(t) -> bool:
    """True when t is an unevaluated lazy graph node (data not yet computed)."""
    return (not getattr(t, "is_leaf", True)
            and getattr(t, "data", None) is None
            and not getattr(t, "on_gpu", False))


def _lazy_gpu_no_grad(t) -> bool:
    """True when a concrete GPU tensor can safely be deferred for fusion."""
    return _use_gpu(t) and not getattr(t, "requires_grad", False)


def _exec_unary(op_name, gpu_fn, cpu_fn, a):
    """Execute a unary op on a concrete tensor and set up autograd if needed."""
    from novax.autograd import attach_unary_grad
    out = gpu_fn(a) if (_use_gpu(a) and gpu_fn) else cpu_fn(a)
    attach_unary_grad(out, a, op_name)
    return out


def _exec_binary(op_name, gpu_fn, cpu_fn, a, b):
    """Execute a binary op on concrete tensors and set up autograd if needed."""
    from novax.autograd import attach_binary_grad
    out = gpu_fn(a, b) if (_use_gpu(a, b) and gpu_fn) else cpu_fn(a, b)
    attach_binary_grad(out, a, b, op_name)
    return out


# ---------------------------------------------------------------------------
# Binary ops
# ---------------------------------------------------------------------------

def add(a, b):
    from novax.core import Tensor
    if _is_lazy(a) or _is_lazy(b):
        return Tensor(None, op="add", inputs=[a, a._wrap(b) if hasattr(a, "_wrap") else b])
    return _exec_binary("add", gpu_add, cpu_add, a, b)

def sub(a, b):
    from novax.core import Tensor
    if _is_lazy(a) or _is_lazy(b):
        return Tensor(None, op="sub", inputs=[a, a._wrap(b) if hasattr(a, "_wrap") else b])
    return _exec_binary("sub", gpu_sub, cpu_sub, a, b)

def mul(a, b):
    from novax.core import Tensor
    if _is_lazy(a) or _is_lazy(b):
        return Tensor(None, op="mul", inputs=[a, a._wrap(b) if hasattr(a, "_wrap") else b])
    return _exec_binary("mul", gpu_mul, cpu_mul, a, b)

def div(a, b):
    from novax.core import Tensor
    if _is_lazy(a) or _is_lazy(b):
        return Tensor(None, op="div", inputs=[a, a._wrap(b) if hasattr(a, "_wrap") else b])
    return _exec_binary("div", gpu_div, cpu_div, a, b)

def pow(a, b):
    from novax.core import Tensor
    if _is_lazy(a) or _is_lazy(b):
        return Tensor(None, op="pow", inputs=[a, a._wrap(b) if hasattr(a, "_wrap") else b])
    return _exec_binary("pow", gpu_pow, cpu_pow, a, b)

def matmul(a, b):
    from novax.core import Tensor
    from novax.autograd import attach_matmul_grad
    if _is_lazy(a) or _is_lazy(b):
        return Tensor(None, op="matmul", inputs=[a, b])
    out = gpu_matmul(a, b) if (_use_gpu(a, b) and gpu_matmul) else cpu_matmul(a, b)
    attach_matmul_grad(out, a, b)
    return out

# ---------------------------------------------------------------------------
# Unary elementwise ops
# ---------------------------------------------------------------------------

def exp(a):
    from novax.core import Tensor
    if _is_lazy(a):
        return Tensor(None, op="exp", inputs=[a])
    return _exec_unary("exp", gpu_exp, cpu_exp, a)

def log(a):
    from novax.core import Tensor
    if _is_lazy(a):
        return Tensor(None, op="log", inputs=[a])
    return _exec_unary("log", gpu_log, cpu_log, a)

def sqrt(a):
    from novax.core import Tensor
    if _is_lazy(a):
        return Tensor(None, op="sqrt", inputs=[a])
    return _exec_unary("sqrt", gpu_sqrt, cpu_sqrt, a)

def abs(a):
    from novax.core import Tensor
    if _is_lazy(a) or _lazy_gpu_no_grad(a):
        return Tensor(None, op="abs", inputs=[a])
    return _exec_unary("abs", gpu_abs, cpu_abs, a)

def neg(a):
    from novax.core import Tensor
    if _is_lazy(a):
        return Tensor(None, op="neg", inputs=[a])
    return _exec_unary("neg", gpu_neg, cpu_neg, a)

# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def relu(a):
    from novax.core import Tensor
    if _is_lazy(a):
        return Tensor(None, op="relu", inputs=[a])
    return _exec_unary("relu", gpu_relu, cpu_relu, a)

def sigmoid(a):
    from novax.core import Tensor
    if _is_lazy(a):
        return Tensor(None, op="sigmoid", inputs=[a])
    return _exec_unary("sigmoid", gpu_sigmoid, cpu_sigmoid, a)

def tanh(a):
    from novax.core import Tensor
    if _is_lazy(a):
        return Tensor(None, op="tanh", inputs=[a])
    return _exec_unary("tanh", gpu_tanh, cpu_tanh, a)

def softmax(a):
    from novax.core import Tensor
    if _is_lazy(a):
        return Tensor(None, op="softmax", inputs=[a])
    return _exec_unary("softmax", gpu_softmax, cpu_softmax, a)

# ---------------------------------------------------------------------------
# Reduction ops
# ---------------------------------------------------------------------------

def sum(a):
    from novax.core import Tensor
    if _is_lazy(a):
        return Tensor(None, op="sum", inputs=[a])
    return _exec_unary("sum", gpu_sum, cpu_sum, a)

def mean(a):
    from novax.core import Tensor
    if _is_lazy(a):
        return Tensor(None, op="mean", inputs=[a])
    return _exec_unary("mean", gpu_mean, cpu_mean, a)

def max(a):
    from novax.core import Tensor
    if _is_lazy(a):
        return Tensor(None, op="max", inputs=[a])
    return _exec_unary("max", gpu_max, cpu_max, a)

def min(a):
    from novax.core import Tensor
    if _is_lazy(a):
        return Tensor(None, op="min", inputs=[a])
    return _exec_unary("min", gpu_min, cpu_min, a)
