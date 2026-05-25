"""
Tests for T2 remaining items (expression memoization, float4 vectorization),
T3 warp-shuffle reductions, and T4 auto CUDA-Graph capture / replay.

CPU-runnable tests verify decision logic and correctness fallbacks.
GPU tests (auto-skipped without CUDA) verify the kernel paths.
"""

import pytest
import numpy as np

import novax as nx
from novax.core import Tensor, GPU_AVAILABLE
from novax.ops import launcher


def _leaf(arr):
    return Tensor(np.asarray(arr, dtype=np.float32))


# ---------------------------------------------------------------------------
# T2 remaining: expression memoization
# ---------------------------------------------------------------------------

class TestFuseTemplateMemoization:
    """_fuse_template caches its result on the node after the first call."""

    def test_cache_populated_after_first_call(self):
        a, b = _leaf([1.0]), _leaf([2.0])
        node = Tensor(None, op="mul", inputs=[a, b])
        assert node._fuse_cache is None
        node._fuse_template()
        assert node._fuse_cache is not None

    def test_second_call_returns_same_object(self):
        a, b = _leaf([1.0]), _leaf([2.0])
        node = Tensor(None, op="add", inputs=[a, b])
        r1 = node._fuse_template()
        r2 = node._fuse_template()
        # Same tuple object returned — Python's `is` check
        assert r1 is r2

    def test_deep_chain_cached(self):
        a, b, c = _leaf([1.0]), _leaf([2.0]), _leaf([3.0])
        mul = Tensor(None, op="mul", inputs=[a, b])
        add = Tensor(None, op="add", inputs=[mul, c])
        relu = Tensor(None, op="relu", inputs=[add])
        relu._fuse_template()
        assert relu._fuse_cache is not None
        # Intermediate nodes don't get cached (only the root is called)
        assert mul._fuse_cache is None   # _fuse_template never called on mul

    def test_cache_survives_repeated_eval(self):
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([2.0,  2.0, 2.0], dtype=np.float32)
        na, nb = _leaf(a), _leaf(b)
        node = Tensor(None, op="relu", inputs=[
            Tensor(None, op="mul", inputs=[na, nb])
        ])
        r1 = node.eval()
        r2 = node.eval()
        np.testing.assert_array_almost_equal(r1.data, r2.data)


# ---------------------------------------------------------------------------
# T3: warp-shuffle reductions — CPU correctness (fallback path)
# ---------------------------------------------------------------------------

class TestReductionsCPU:
    """CPU eval of sum/mean/max/min must match numpy."""

    def test_sum(self):
        arr = np.arange(1, 11, dtype=np.float32)
        result = nx.sum(_leaf(arr)).eval()
        np.testing.assert_almost_equal(result.data[0], np.sum(arr), decimal=5)

    def test_mean(self):
        arr = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        result = nx.mean(_leaf(arr)).eval()
        np.testing.assert_almost_equal(result.data[0], np.mean(arr), decimal=5)

    def test_max(self):
        arr = np.array([-1.0, 5.0, 3.0, -2.0], dtype=np.float32)
        result = nx.max(_leaf(arr)).eval()
        np.testing.assert_almost_equal(result.data[0], np.max(arr), decimal=5)

    def test_min(self):
        arr = np.array([4.0, -3.0, 7.0, 0.0], dtype=np.float32)
        result = nx.min(_leaf(arr)).eval()
        np.testing.assert_almost_equal(result.data[0], np.min(arr), decimal=5)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestReductionsGPU:
    """Warp-shuffle reductions on GPU match numpy for various sizes."""

    def test_sum_small(self):
        arr = np.arange(1, 33, dtype=np.float32)   # 32 elements = one warp
        t = Tensor(arr).to_gpu()
        result = nx.sum(t).eval()
        np.testing.assert_almost_equal(result.to_host()[0], np.sum(arr), decimal=3)

    def test_sum_multi_block(self):
        np.random.seed(0)
        arr = np.random.randn(100_000).astype(np.float32)
        t = Tensor(arr).to_gpu()
        result = nx.sum(t).eval()
        np.testing.assert_almost_equal(result.to_host()[0], np.sum(arr), decimal=0)

    def test_mean_gpu(self):
        np.random.seed(1)
        arr = np.random.randn(8192).astype(np.float32)
        t = Tensor(arr).to_gpu()
        result = nx.mean(t).eval()
        np.testing.assert_almost_equal(result.to_host()[0], np.mean(arr), decimal=3)

    def test_max_gpu(self):
        arr = np.array([-5.0, 1.0, 9.0, 3.0, -1.0] * 1000, dtype=np.float32)
        t = Tensor(arr).to_gpu()
        result = nx.max(t).eval()
        np.testing.assert_almost_equal(result.to_host()[0], 9.0, decimal=4)

    def test_min_gpu(self):
        arr = np.array([4.0, -7.0, 2.0, 0.0] * 512, dtype=np.float32)
        t = Tensor(arr).to_gpu()
        result = nx.min(t).eval()
        np.testing.assert_almost_equal(result.to_host()[0], -7.0, decimal=4)

    def test_sum_power_of_two(self):
        arr = np.ones(2 ** 20, dtype=np.float32)
        t = Tensor(arr).to_gpu()
        result = nx.sum(t).eval()
        np.testing.assert_almost_equal(result.to_host()[0], 2 ** 20, decimal=0)


# ---------------------------------------------------------------------------
# T2 float4: vectorized fused kernel source generation
# ---------------------------------------------------------------------------

class TestFloat4KernelSource:
    """_build_fused_v4_src generates valid per-component substitutions."""

    def test_single_input_relu_expr(self):
        src = launcher._build_fused_v4_src("test_v4", 1, "fmaxf(0.0f, x0[idx])")
        assert "v0.x" in src
        assert "v0.y" in src
        assert "v0.z" in src
        assert "v0.w" in src
        assert "x0[idx]" not in src     # fully substituted

    def test_two_input_mul_expr(self):
        src = launcher._build_fused_v4_src("mul_v4", 2, "(x0[idx] * x1[idx])")
        assert "v0.x * v1.x" in src
        assert "v0.w * v1.w" in src

    def test_no_idx_leakage(self):
        src = launcher._build_fused_v4_src("relu_v4", 1, "fmaxf(0.0f, x0[idx])")
        # After substitution, bare "[idx]" should not appear in vout lines
        for line in src.split("\n"):
            if "vout." in line:
                assert "[idx]" not in line


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestFloat4GPU:
    """Float4 vectorized path produces correct results for power-of-4 sizes."""

    def test_relu_vec4(self):
        np.random.seed(2)
        arr = np.random.randn(4096).astype(np.float32)
        t = Tensor(arr).to_gpu()
        node = nx.relu(t)
        result = node._try_full_fuse()
        assert result is not None
        np.testing.assert_array_almost_equal(result.to_host(), np.maximum(0.0, arr), decimal=4)

    def test_mul_add_vec4(self):
        np.random.seed(3)
        a_np = np.random.randn(8192).astype(np.float32)
        b_np = np.random.randn(8192).astype(np.float32)
        c_np = np.random.randn(8192).astype(np.float32)
        a = Tensor(a_np).to_gpu()
        b = Tensor(b_np).to_gpu()
        c = Tensor(c_np).to_gpu()
        result = (a * b + c).eval()
        np.testing.assert_array_almost_equal(result.to_host(), a_np * b_np + c_np, decimal=4)


# ---------------------------------------------------------------------------
# T4: CUDA Graph auto-capture and replay
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestCUDAGraphReplay:
    """Repeated eval with identical input pointers replays a cached graph."""

    def test_repeated_eval_identical_results(self):
        np.random.seed(10)
        a_np = np.random.randn(4096).astype(np.float32)
        b_np = np.random.randn(4096).astype(np.float32)
        c_np = np.random.randn(4096).astype(np.float32)
        a = Tensor(a_np).to_gpu()
        b = Tensor(b_np).to_gpu()
        c = Tensor(c_np).to_gpu()
        expr = nx.tanh(nx.relu(a * b) + c)
        ref = np.tanh(np.maximum(0.0, a_np * b_np) + c_np)

        results = [expr.eval().to_host() for _ in range(5)]
        for r in results:
            np.testing.assert_array_almost_equal(r, ref, decimal=4)

    def test_six_op_chain_repeated(self):
        np.random.seed(11)
        a_np = np.random.randn(2048).astype(np.float32)
        b_np = np.random.randn(2048).astype(np.float32)
        c_np = np.random.randn(2048).astype(np.float32)
        a = Tensor(a_np).to_gpu()
        b = Tensor(b_np).to_gpu()
        c = Tensor(c_np).to_gpu()
        expr = nx.sigmoid(nx.exp(a) + nx.relu(b * c))
        ref = 1.0 / (1.0 + np.exp(-(np.exp(a_np) + np.maximum(0.0, b_np * c_np))))

        r1 = expr.eval().to_host()
        r2 = expr.eval().to_host()
        r3 = expr.eval().to_host()
        np.testing.assert_array_almost_equal(r1, ref, decimal=4)
        np.testing.assert_array_almost_equal(r2, ref, decimal=4)
        np.testing.assert_array_almost_equal(r3, ref, decimal=4)

    def test_repeated_fused_eval_stable(self):
        """Repeated fused eval with stable input pointers yields identical results."""
        np.random.seed(12)
        a_np = np.random.randn(1024).astype(np.float32)
        b_np = np.random.randn(1024).astype(np.float32)
        a = Tensor(a_np).to_gpu()
        b = Tensor(b_np).to_gpu()
        ref = np.maximum(0.0, a_np * b_np)
        r1 = nx.relu(a * b).eval().to_host()
        r2 = nx.relu(a * b).eval().to_host()
        np.testing.assert_array_almost_equal(r1, ref, decimal=4)
        np.testing.assert_array_equal(r1, r2)

    def test_mlp_forward_repeated(self):
        """Three consecutive MLP forward passes produce the same scalar loss."""
        np.random.seed(13)
        BS, IN, HID, OUT = 32, 16, 24, 8
        X_np  = np.random.randn(BS, IN).astype(np.float32)
        W1_np = np.random.randn(IN, HID).astype(np.float32) * 0.1
        b1_np = np.random.randn(HID).astype(np.float32)
        W2_np = np.random.randn(HID, OUT).astype(np.float32) * 0.1
        b2_np = np.random.randn(OUT).astype(np.float32)

        X  = Tensor(X_np).to_gpu()
        W1 = Tensor(W1_np).to_gpu()
        b1 = Tensor(b1_np).to_gpu()
        W2 = Tensor(W2_np).to_gpu()
        b2 = Tensor(b2_np).to_gpu()

        def fwd():
            h = nx.relu(nx.matmul(X, W1) + b1)
            return nx.mean(nx.matmul(h, W2) + b2).eval()

        h_ref = np.maximum(0.0, X_np @ W1_np + b1_np)
        ref   = np.mean(h_ref @ W2_np + b2_np)

        for _ in range(3):
            loss = fwd()
            np.testing.assert_almost_equal(loss.to_host()[0], ref, decimal=3)
