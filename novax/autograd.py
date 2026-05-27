import numpy as np

_grad_enabled = True


def _get_grad_enabled() -> bool:
    return _grad_enabled


class no_grad:
    """Context manager that disables gradient tracking within its scope."""

    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self._prev


# ---------------------------------------------------------------------------
# Shared backward-closure setup utilities
# Used by both core.Tensor.eval() and dispatch functions.
# ---------------------------------------------------------------------------

def attach_unary_grad(out, inp, op: str):
    """Attach a backward closure for a unary op result."""
    from novax.core import Tensor  # local import avoids circular import
    if not _grad_enabled or not getattr(inp, "requires_grad", False):
        return
    out.requires_grad = True
    out._prev = {inp}
    inp_shape = inp.shape
    inp_size = inp.size
    needs_value = op not in ("sum", "mean")
    inp_data = (inp.to_host() if inp.on_gpu else inp.data) if needs_value else None

    def _bwd(inp=inp, inp_data=inp_data, inp_shape=inp_shape, inp_size=inp_size, out=out, op=op):
        g = out.grad.data if out.grad is not None else np.ones(out.shape, dtype=np.float32)
        if op == "neg":
            dL = -g
        elif op == "exp":
            out_arr = out.to_host() if out.on_gpu else out.data
            dL = g * out_arr
        elif op == "log":
            dL = g / inp_data
        elif op == "sqrt":
            dL = g / (2.0 * np.sqrt(np.maximum(inp_data, 1e-12)))
        elif op == "abs":
            dL = g * np.sign(inp_data)
        elif op == "relu":
            dL = g * (inp_data > 0).astype(np.float32)
        elif op == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-inp_data))
            dL = g * s * (1.0 - s)
        elif op == "tanh":
            t = np.tanh(inp_data)
            dL = g * (1.0 - t ** 2)
        elif op == "sum":
            dL = np.full(inp_shape, float(g.flat[0]), dtype=np.float32)
        elif op == "mean":
            dL = np.full(inp_shape, float(g.flat[0]) / float(inp_size), dtype=np.float32)
        elif op == "max":
            mask = (inp_data == np.max(inp_data)).astype(np.float32)
            mask /= np.sum(mask)
            dL = float(g.flat[0]) * mask
        elif op == "min":
            mask = (inp_data == np.min(inp_data)).astype(np.float32)
            mask /= np.sum(mask)
            dL = float(g.flat[0]) * mask
        else:
            dL = np.zeros_like(inp_data)
        acc = inp.grad.data + dL if inp.grad is not None else dL.copy()
        inp.grad = Tensor(acc.astype(np.float32))

    out._backward = _bwd


def _unbroadcast(g: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Sum gradient axes that were broadcast to reach g's shape."""
    while g.ndim > len(target_shape):
        g = g.sum(axis=0)
    for i, (gs, ts) in enumerate(zip(g.shape, target_shape)):
        if ts == 1 and gs != 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(target_shape)


def attach_binary_grad(out, left, right, op: str):
    """Attach a backward closure for a binary elementwise op result."""
    from novax.core import Tensor
    needs = _grad_enabled and (
        getattr(left, "requires_grad", False) or getattr(right, "requires_grad", False)
    )
    if not needs:
        return
    out.requires_grad = True
    out._prev = set()
    if getattr(left, "requires_grad", False):
        out._prev.add(left)
    if getattr(right, "requires_grad", False):
        out._prev.add(right)

    left_shape = left.shape
    right_shape = right.shape
    needs_values = op not in ("add", "sub")
    left_arr = (left.to_host() if left.on_gpu else left.data) if needs_values else None
    right_arr = (right.to_host() if right.on_gpu else right.data) if needs_values else None

    def _bwd(
        left=left,
        right=right,
        left_arr=left_arr,
        right_arr=right_arr,
        left_shape=left_shape,
        right_shape=right_shape,
        out=out,
        op=op,
    ):
        g = out.grad.data if out.grad is not None else np.ones(out.shape, dtype=np.float32)
        if op == "add":
            dL, dR = g, g
        elif op == "sub":
            dL, dR = g, -g
        elif op == "mul":
            dL = g * right_arr
            dR = g * left_arr
        elif op == "div":
            dL = g / right_arr
            dR = -g * left_arr / (right_arr ** 2)
        elif op == "pow":
            dL = g * right_arr * np.power(np.abs(left_arr) + 1e-12, right_arr - 1)
            dR = g * np.power(np.abs(left_arr) + 1e-12, right_arr) * np.log(np.abs(left_arr) + 1e-12)
        else:
            dL = np.zeros(left_shape, dtype=np.float32)
            dR = np.zeros(right_shape, dtype=np.float32)

        if getattr(left, "requires_grad", False):
            dL = _unbroadcast(np.array(dL, dtype=np.float32), left_shape)
            acc = left.grad.data + dL if left.grad is not None else dL.copy()
            left.grad = Tensor(acc.astype(np.float32))
        if getattr(right, "requires_grad", False):
            dR = _unbroadcast(np.array(dR, dtype=np.float32), right_shape)
            acc = right.grad.data + dR if right.grad is not None else dR.copy()
            right.grad = Tensor(acc.astype(np.float32))

    out._backward = _bwd


def attach_matmul_grad(out, left, right):
    """Attach a backward closure for a matmul result."""
    from novax.core import Tensor
    needs = _grad_enabled and (
        getattr(left, "requires_grad", False) or getattr(right, "requires_grad", False)
    )
    if not needs:
        return
    out.requires_grad = True
    out._prev = set()
    if getattr(left, "requires_grad", False):
        out._prev.add(left)
    if getattr(right, "requires_grad", False):
        out._prev.add(right)

    left_arr = (left.to_host() if left.on_gpu else left.data) if getattr(right, "requires_grad", False) else None
    right_arr = (right.to_host() if right.on_gpu else right.data) if getattr(left, "requires_grad", False) else None

    def _bwd(left=left, right=right, left_arr=left_arr, right_arr=right_arr, out=out):
        g = out.grad.data if out.grad is not None else np.ones(out.shape, dtype=np.float32)
        if getattr(left, "requires_grad", False):
            dL = g @ right_arr.T
            acc = left.grad.data + dL if left.grad is not None else dL.copy()
            left.grad = Tensor(acc.astype(np.float32))
        if getattr(right, "requires_grad", False):
            dR = left_arr.T @ g
            acc = right.grad.data + dR if right.grad is not None else dR.copy()
            right.grad = Tensor(acc.astype(np.float32))

    out._backward = _bwd
