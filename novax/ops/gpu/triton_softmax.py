"""
Item 7: Triton fused 1D softmax.

Replaces the 5-pass PyCUDA implementation (max → shift → exp → sum → div)
with a single Triton kernel that keeps data in registers throughout.

Falls back to None on import error or when n > MAX_TRITON_N, signalling the
caller to use the multi-pass PyCUDA fallback.
"""

import numpy as np

MAX_TRITON_N = 65536   # max elements handled by the single-block kernel

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def _softmax_fwd_1d(
        X_ptr, Y_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Single-block fused softmax for 1D inputs up to BLOCK_SIZE elements."""
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(X_ptr + offsets, mask=mask, other=-float("inf"))

        # Numerically stable: subtract max before exp
        x_max = tl.max(x, axis=0)
        x = x - x_max
        exp_x = tl.exp(x)
        inv_sum = 1.0 / tl.sum(exp_x, axis=0)
        y = exp_x * inv_sum

        tl.store(Y_ptr + offsets, y, mask=mask)


def triton_softmax_1d(a):
    """
    Fused 1D softmax via Triton.

    Parameters
    ----------
    a : Tensor  (on GPU, float32)

    Returns
    -------
    Tensor on GPU, or None if Triton is unavailable / n too large.
    """
    if not _TRITON_AVAILABLE:
        return None

    n = a.size
    if n > MAX_TRITON_N:
        return None   # signal caller to use multi-pass fallback

    try:
        import torch
        import pycuda.driver as cuda
        from novax.utils import mempool

        # Round up to next power of 2 for the Triton constexpr BLOCK_SIZE
        BLOCK = triton.next_power_of_2(max(n, 1))

        # Bridge PyCUDA ↔ Triton via PyTorch CUDA tensors.
        # Both live on CUDA device 0 / same context, so dtod copies stay in VRAM.
        x_torch = torch.empty(n, dtype=torch.float32, device="cuda")
        y_torch = torch.empty(n, dtype=torch.float32, device="cuda")

        # Copy from PyCUDA buffer into PyTorch tensor
        cuda.memcpy_dtod(x_torch.data_ptr(), a.gpu_ptr, n * 4)

        # Launch Triton kernel
        grid = (1,)
        _softmax_fwd_1d[grid](x_torch, y_torch, n, BLOCK_SIZE=BLOCK)
        torch.cuda.synchronize()

        # Copy result back to a PyCUDA buffer (stays in VRAM)
        out_ptr = mempool.alloc(n * 4)
        cuda.memcpy_dtod(out_ptr, y_torch.data_ptr(), n * 4)

        from importlib import import_module
        Tensor = getattr(import_module("novax.core"), "Tensor")
        result = Tensor(out_ptr, gpu=True, inputs=[a])
        result.shape = a.shape
        result.size = n
        result.dtype = np.float32
        return result

    except Exception:
        return None   # any failure → multi-pass fallback
