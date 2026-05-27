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

def _accumulate_grad(t, grad):
    from novax.core import Tensor
    if t.grad is None:
        t.grad = grad
        return
    if getattr(t.grad, "on_gpu", False) and getattr(grad, "on_gpu", False):
        from novax.ops.launcher import launch_kernel
        t.grad = launch_kernel(t.grad, grad, "grad_accum_add_kernel", "a[idx] + b[idx]")
        return
    current = t.grad.to_host() if getattr(t.grad, "on_gpu", False) else t.grad.data
    incoming = grad.to_host() if getattr(grad, "on_gpu", False) else grad.data
    t.grad = Tensor((current + incoming).astype(np.float32))


def attach_unary_grad(out, inp, op: str):
    """Attach a backward closure for a unary op result."""
    from novax.core import Tensor  # local import avoids circular import
    if not _grad_enabled or not getattr(inp, "requires_grad", False):
        return
    out.requires_grad = True
    out._prev = {inp}
    gpu_backward = getattr(inp, "on_gpu", False) and op in ("sum", "mean", "relu")
    inp_data = None if gpu_backward else (inp.to_host() if inp.on_gpu else inp.data)

    def _bwd(inp=inp, inp_data=inp_data, out=out, op=op):
        grad = out.grad
        if grad is not None and getattr(grad, "on_gpu", False) and getattr(inp, "on_gpu", False):
            if op in ("sum", "mean"):
                from novax.ops.launcher import launch_fill_from_scalar
                scale = 1.0 if op == "sum" else 1.0 / float(inp.size)
                _accumulate_grad(inp, launch_fill_from_scalar(grad, inp.shape, scale=scale))
                return
            if op == "relu":
                from novax.ops.launcher import launch_relu_backward
                _accumulate_grad(inp, launch_relu_backward(inp, grad))
                return

        g = (grad.to_host() if getattr(grad, "on_gpu", False) else grad.data) if grad is not None else np.ones(out.shape, dtype=np.float32)
        if inp_data is None and op not in ("sum", "mean"):
            inp_data = inp.to_host() if inp.on_gpu else inp.data
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
            dL = np.full(inp.shape, float(g.flat[0]), dtype=np.float32)
        elif op == "mean":
            dL = np.full(inp.shape, float(g.flat[0]) / float(inp.size), dtype=np.float32)
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
        _accumulate_grad(inp, Tensor(dL.astype(np.float32)))

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

    gpu_add_sub = (
        op in ("add", "sub")
        and getattr(left, "on_gpu", False)
        and getattr(right, "on_gpu", False)
    )
    left_arr = None if gpu_add_sub else (left.to_host() if left.on_gpu else left.data)
    right_arr = None if gpu_add_sub else (right.to_host() if right.on_gpu else right.data)

    def _bwd(left=left, right=right, left_arr=left_arr, right_arr=right_arr, out=out, op=op):
        grad = out.grad
        if (
            grad is not None
            and getattr(grad, "on_gpu", False)
            and op in ("add", "sub")
            and getattr(left, "on_gpu", False)
            and getattr(right, "on_gpu", False)
        ):
            from novax.ops.launcher import launch_kernel, launch_reduce_axis0
            if getattr(left, "requires_grad", False):
                if left.shape == grad.shape:
                    dL = grad
                elif len(grad.shape) == 2 and left.shape == (grad.shape[1],):
                    dL = launch_reduce_axis0(grad)
                else:
                    dL = Tensor(_unbroadcast(grad.to_host(), left.shape))
                _accumulate_grad(left, dL)
            if getattr(right, "requires_grad", False):
                sign = 1.0 if op == "add" else -1.0
                if right.shape == grad.shape:
                    dR = grad if sign > 0 else launch_kernel(grad, None, "grad_neg_kernel", "(-a[idx])")
                elif len(grad.shape) == 2 and right.shape == (grad.shape[1],):
                    dR = launch_reduce_axis0(grad, scale=sign)
                else:
                    g_host = grad.to_host()
                    dR = Tensor(_unbroadcast(np.array(sign * g_host, dtype=np.float32), right.shape))
                _accumulate_grad(right, dR)
            return

        left_data = left_arr if left_arr is not None else (left.to_host() if left.on_gpu else left.data)
        right_data = right_arr if right_arr is not None else (right.to_host() if right.on_gpu else right.data)
        g = (grad.to_host() if getattr(grad, "on_gpu", False) else grad.data) if grad is not None else np.ones(out.shape, dtype=np.float32)
        if op == "add":
            dL, dR = g, g
        elif op == "sub":
            dL, dR = g, -g
        elif op == "mul":
            dL = g * right_data
            dR = g * left_data
        elif op == "div":
            dL = g / right_data
            dR = -g * left_data / (right_data ** 2)
        elif op == "pow":
            dL = g * right_data * np.power(np.abs(left_data) + 1e-12, right_data - 1)
            dR = g * np.power(np.abs(left_data) + 1e-12, right_data) * np.log(np.abs(left_data) + 1e-12)
        else:
            dL = dR = np.zeros_like(left_data)

        if getattr(left, "requires_grad", False):
            dL = _unbroadcast(np.array(dL, dtype=np.float32), left_data.shape)
            _accumulate_grad(left, Tensor(dL.astype(np.float32)))
        if getattr(right, "requires_grad", False):
            dR = _unbroadcast(np.array(dR, dtype=np.float32), right_data.shape)
            _accumulate_grad(right, Tensor(dR.astype(np.float32)))

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

    gpu_backward = getattr(left, "on_gpu", False) and getattr(right, "on_gpu", False)
    left_arr = None if gpu_backward else (left.to_host() if left.on_gpu else left.data)
    right_arr = None if gpu_backward else (right.to_host() if right.on_gpu else right.data)

    def _bwd(left=left, right=right, left_arr=left_arr, right_arr=right_arr, out=out):
        grad = out.grad
        if (
            grad is not None
            and getattr(grad, "on_gpu", False)
            and getattr(left, "on_gpu", False)
            and getattr(right, "on_gpu", False)
        ):
            from novax.ops.launcher import launch_matmul_transpose
            if getattr(left, "requires_grad", False):
                _accumulate_grad(left, launch_matmul_transpose(grad, right, False, True))
            if getattr(right, "requires_grad", False):
                _accumulate_grad(right, launch_matmul_transpose(left, grad, True, False))
            return

        left_data = left_arr if left_arr is not None else (left.to_host() if left.on_gpu else left.data)
        right_data = right_arr if right_arr is not None else (right.to_host() if right.on_gpu else right.data)
        g = (grad.to_host() if getattr(grad, "on_gpu", False) else grad.data) if grad is not None else np.ones(out.shape, dtype=np.float32)
        if getattr(left, "requires_grad", False):
            dL = g @ right_data.T
            _accumulate_grad(left, Tensor(dL.astype(np.float32)))
        if getattr(right, "requires_grad", False):
            dR = left_data.T @ g
            _accumulate_grad(right, Tensor(dR.astype(np.float32)))

    out._backward = _bwd
