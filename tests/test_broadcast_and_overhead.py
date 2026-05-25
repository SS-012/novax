"""
Tests for Tier 0 (broadcasting in the forward GPU binary path) and
Tier 1 (per-launch CPU overhead removal) of the GPU optimization plan.

Tier 0 fixes the "All inputs must have same size" crash that hit the MLP
forward / inference benchmarks: `matmul(M,N) + bias(N,)` is a trailing-dim
broadcast that the fused elementwise path could not express.

Tier 1 removes per-launch driver round-trips: the block-size query, the
stream lookup, and cuBLAS handle creation are all cached.

CPU-runnable tests verify decision logic and numpy-matching correctness.
GPU-only tests (skipped without CUDA) verify the broadcast kernel itself.
"""

import pytest
import numpy as np

import novax as nx
from novax.core import Tensor, GPU_AVAILABLE
from novax.ops import launcher


# ---------------------------------------------------------------------------
# Tier 0: trailing-dim broadcast decision logic
# ---------------------------------------------------------------------------

class TestTrailingBroadcastHelper:
    """`Tensor._is_trailing_broadcast` decides if `small` broadcasts on `big`."""

    def test_row_vector_bias_matches(self):
        assert Tensor._is_trailing_broadcast((8, 4), (4,), 4) is True

    def test_three_d_trailing_matches(self):
        assert Tensor._is_trailing_broadcast((2, 3, 4), (3, 4), 12) is True
        assert Tensor._is_trailing_broadcast((2, 3, 4), (4,), 4) is True

    def test_scalar_always_broadcasts(self):
        assert Tensor._is_trailing_broadcast((8, 4), (1,), 1) is True

    def test_leading_mismatch_rejected(self):
        # (8,4) + (8,) is NOT a valid trailing broadcast (would need (8,1))
        assert Tensor._is_trailing_broadcast((8, 4), (8,), 8) is False

    def test_small_rank_exceeds_big_rejected(self):
        assert Tensor._is_trailing_broadcast((4,), (2, 4), 8) is False

    def test_none_shapes_rejected(self):
        assert Tensor._is_trailing_broadcast(None, (4,), 4) is False
        assert Tensor._is_trailing_broadcast((8, 4), None, 4) is False


# ---------------------------------------------------------------------------
# Tier 0: matmul + bias correctness (the previously-crashing pattern)
# ---------------------------------------------------------------------------

class TestMatmulBiasCPU:
    """CPU eval of `matmul + bias` must match numpy (regression lock)."""

    def test_matmul_plus_bias(self):
        np.random.seed(0)
        M, K, N = 8, 4, 6
        X = np.random.randn(M, K).astype(np.float32)
        W = np.random.randn(K, N).astype(np.float32)
        b = np.random.randn(N).astype(np.float32)
        out = (nx.matmul(Tensor(X), Tensor(W)) + Tensor(b)).eval()
        np.testing.assert_array_almost_equal(out.data, X @ W + b, decimal=5)

    def test_bias_plus_matmul_order(self):
        np.random.seed(1)
        M, K, N = 5, 3, 7
        X = np.random.randn(M, K).astype(np.float32)
        W = np.random.randn(K, N).astype(np.float32)
        b = np.random.randn(N).astype(np.float32)
        out = (Tensor(b) + nx.matmul(Tensor(X), Tensor(W))).eval()
        np.testing.assert_array_almost_equal(out.data, b + X @ W, decimal=5)

    def test_mlp_forward_structure(self):
        """Mirrors notebook cells 8/13 — previously crashed with size mismatch."""
        np.random.seed(2)
        BS, IN, HID, OUT = 16, 8, 12, 4
        X = np.random.randn(BS, IN).astype(np.float32)
        W1 = np.random.randn(IN, HID).astype(np.float32) * 0.1
        b1 = np.random.randn(HID).astype(np.float32)
        W2 = np.random.randn(HID, OUT).astype(np.float32) * 0.1
        b2 = np.random.randn(OUT).astype(np.float32)

        h = nx.relu(nx.matmul(Tensor(X), Tensor(W1)) + Tensor(b1))
        out = nx.mean(nx.matmul(h, Tensor(W2)) + Tensor(b2)).eval()

        h_ref = np.maximum(0.0, X @ W1 + b1)
        ref = np.mean(h_ref @ W2 + b2)
        np.testing.assert_array_almost_equal(out.data, [ref], decimal=4)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestBroadcastBinaryGPU:
    """GPU broadcast kernel and the matmul+bias eval path."""

    def test_launch_broadcast_binary_direct(self):
        M, N = 6, 5
        A = np.random.randn(M, N).astype(np.float32)
        b = np.random.randn(N).astype(np.float32)
        big = Tensor(A).to_gpu()
        small = Tensor(b).to_gpu()
        out = launcher.launch_broadcast_binary(big, small, "+", False, "add_bcast_r_kernel")
        np.testing.assert_array_almost_equal(out.to_host(), A + b, decimal=4)

    def test_matmul_plus_bias_gpu(self):
        np.random.seed(3)
        M, K, N = 16, 8, 6
        X = np.random.randn(M, K).astype(np.float32)
        W = np.random.randn(K, N).astype(np.float32)
        b = np.random.randn(N).astype(np.float32)
        Xn = Tensor(X).to_gpu(); Wn = Tensor(W).to_gpu(); bn = Tensor(b).to_gpu()
        out = (nx.matmul(Xn, Wn) + bn).eval()
        np.testing.assert_array_almost_equal(out.to_host(), X @ W + b, decimal=4)

    def test_sub_broadcast_order_preserved(self):
        M, N = 4, 5
        A = np.random.randn(M, N).astype(np.float32)
        b = np.random.randn(N).astype(np.float32)
        big = Tensor(A).to_gpu()
        small = Tensor(b).to_gpu()
        # small_is_left=True → b[col] - a[idx]
        out = launcher.launch_broadcast_binary(big, small, "-", True, "sub_bcast_l_kernel")
        np.testing.assert_array_almost_equal(out.to_host(), b - A, decimal=4)


# ---------------------------------------------------------------------------
# Tier 1: per-launch overhead caching
# ---------------------------------------------------------------------------

class TestLaunchOverheadCaching:
    """Block size, stream and cuBLAS handle are resolved at most once."""

    def test_block_size_cached(self):
        first = launcher._optimal_block_size()
        second = launcher._optimal_block_size()
        assert first == second
        # The expensive device query is only cached on the GPU path.
        if launcher.cuda is not None:
            assert launcher._BLOCK_SIZE == first

    def test_block_size_value_valid(self):
        assert launcher._optimal_block_size() in (128, 256, 512)

    def test_get_stream_no_context_returns_none(self):
        # On a CPU-only box (cuda is None) there is no stream.
        if launcher.cuda is None:
            assert launcher._get_stream() is None

    def test_cublas_handle_graceful_without_skcuda(self):
        # Without scikit-cuda installed, handle resolution returns None cleanly.
        try:
            import skcuda  # noqa: F401
            has_skcuda = True
        except Exception:
            has_skcuda = False
        if not has_skcuda:
            assert launcher._get_cublas_handle() is None

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_cublas_handle_is_persistent(self):
        h1 = launcher._get_cublas_handle()
        h2 = launcher._get_cublas_handle()
        if h1 is not None:
            assert h1 == h2  # same handle reused, not recreated
