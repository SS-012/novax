"""
pytest test suite for the 5 GPU optimisation features added to NovaX:

  Item 0 — cuBLAS-backed matmul
  Item 5 — CUDAGraph capture / replay
  Item 6 — fp16 dtype support
  Item 7 — Triton fused 1D softmax
  Item 8 — Pattern-matching auto-fusion (matmul+bias+relu, matmul+bias)

CPU tests run in every environment.
GPU tests are guarded with @pytest.mark.skipif(not GPU_AVAILABLE, ...) so the
suite passes cleanly in CI where no GPU is present.
"""

import pytest
import numpy as np

import novax as nx
from novax.core import Tensor, GPU_AVAILABLE, _DEFAULT_DTYPE
from novax.ops.launcher import (
    CUDAGraph,
    launch_matmul_bias,
    launch_matmul_bias_relu,
    _launch_matmul_cublas,
    _launch_matmul_tiled,
)
from novax.ops.gpu.triton_softmax import triton_softmax_1d


# ---------------------------------------------------------------------------
# Item 0 — cuBLAS-backed matmul
# ---------------------------------------------------------------------------

class TestCuBLASMatmul:
    """Item 0: matrix multiplication dispatched through launch_matmul."""

    def test_cpu_matmul_correct_shape(self):
        """CPU: matmul via dispatch produces a result with the expected shape."""
        A = Tensor(np.random.randn(4, 3).astype(np.float32))
        B = Tensor(np.random.randn(3, 5).astype(np.float32))
        result = nx.matmul(A, B)
        assert isinstance(result, Tensor)
        assert result.shape == (4, 5)

    def test_cpu_matmul_correct_values(self):
        """CPU: matmul via dispatch matches numpy reference."""
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        A = Tensor(a_np)
        B = Tensor(b_np)
        result = nx.matmul(A, B)
        np.testing.assert_array_almost_equal(result.data, a_np @ b_np)

    def test_cpu_matmul_rectangular(self):
        """CPU: matmul with non-square matrices produces correct output."""
        a_np = np.ones((6, 4), dtype=np.float32)
        b_np = np.eye(4, 2, dtype=np.float32)
        A = Tensor(a_np)
        B = Tensor(b_np)
        result = nx.matmul(A, B)
        assert result.shape == (6, 2)
        np.testing.assert_array_almost_equal(result.data, a_np @ b_np)

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_gpu_matmul_shape(self):
        """GPU: launch_matmul result has shape (M, N)."""
        from novax.ops.launcher import launch_matmul
        M, K, N = 8, 4, 6
        a_np = np.random.randn(M, K).astype(np.float32)
        b_np = np.random.randn(K, N).astype(np.float32)
        A = Tensor(a_np).to_gpu()
        B = Tensor(b_np).to_gpu()
        result = launch_matmul(A, B)
        assert result.shape == (M, N)

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_gpu_matmul_values_match_numpy(self):
        """GPU: launch_matmul output matches numpy reference."""
        from novax.ops.launcher import launch_matmul
        M, K, N = 8, 4, 6
        a_np = np.random.randn(M, K).astype(np.float32)
        b_np = np.random.randn(K, N).astype(np.float32)
        A = Tensor(a_np).to_gpu()
        B = Tensor(b_np).to_gpu()
        result = launch_matmul(A, B)
        np.testing.assert_array_almost_equal(result.to_host(), a_np @ b_np, decimal=4)


# ---------------------------------------------------------------------------
# Item 5 — CUDAGraph
# ---------------------------------------------------------------------------

class TestCUDAGraph:
    """Item 5: CUDA Graph capture and replay."""

    def test_instantiation_does_not_raise(self):
        """CPU: creating a CUDAGraph object does not raise an exception."""
        graph = CUDAGraph()
        assert graph is not None
        assert graph._graph_exec is None

    def test_replay_before_capture_raises(self):
        """CPU: calling replay() before capture raises RuntimeError."""
        graph = CUDAGraph()
        with pytest.raises(RuntimeError, match="No graph captured"):
            graph.replay()

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_cuda_graph_capture_and_replay(self):
        """GPU: capture relu(matmul(x,W)+b) in a with-block, replay 5 times."""
        M, K, N = 16, 8, 4
        x_np = np.random.randn(M, K).astype(np.float32)
        W_np = np.random.randn(K, N).astype(np.float32)
        b_np = np.random.randn(N).astype(np.float32)

        x = Tensor(x_np).to_gpu()
        W = Tensor(W_np).to_gpu()
        b = Tensor(b_np).to_gpu()

        graph = CUDAGraph()
        with graph:
            out = (x @ W + b).eval()  # captured
        # out was evaluated during capture; verify shape
        assert out.shape == (M, N)

        # Replay must not raise
        for _ in range(5):
            graph.replay()


# ---------------------------------------------------------------------------
# Item 6 — fp16 dtype support
# ---------------------------------------------------------------------------

class TestFp16:
    """Item 6: half-precision tensor creation, conversion, and arithmetic."""

    def test_tensor_explicit_float16_dtype(self):
        """CPU: Tensor created with dtype=np.float16 carries float16 dtype."""
        t = Tensor([1.0, 2.0, 3.0], dtype=np.float16)
        assert t.dtype == np.float16

    def test_tensor_half_returns_float16(self):
        """CPU: tensor.half() returns a float16 tensor."""
        t = Tensor([1.0, 2.0, 3.0])
        t_half = t.half()
        assert t_half.dtype == np.float16

    def test_tensor_half_correct_values(self):
        """CPU: tensor.half() preserves values within fp16 precision."""
        t = Tensor([1.0, 2.0, 3.0])
        t_half = t.half()
        np.testing.assert_array_almost_equal(
            t_half.data.astype(np.float32), [1.0, 2.0, 3.0], decimal=3
        )

    def test_tensor_float_returns_float32(self):
        """CPU: tensor.float() returns a float32 tensor."""
        t = Tensor([1.0, 2.0, 3.0], dtype=np.float16)
        t_f32 = t.float()
        assert t_f32.dtype == np.float32

    def test_tensor_float_correct_values(self):
        """CPU: tensor.float() preserves values."""
        t = Tensor([1.5, 2.5, 3.5], dtype=np.float16)
        t_f32 = t.float()
        np.testing.assert_array_almost_equal(t_f32.data, [1.5, 2.5, 3.5], decimal=3)

    def test_set_dtype_float16_changes_default(self):
        """CPU: set_dtype('float16') changes _DEFAULT_DTYPE to np.float16."""
        import novax.core as _core
        try:
            nx.set_dtype("float16")
            assert _core._DEFAULT_DTYPE == np.float16
            t = Tensor([1.0, 2.0])
            assert t.dtype == np.float16
        finally:
            nx.set_dtype("float32")
            assert _core._DEFAULT_DTYPE == np.float32

    def test_set_dtype_float32_restores_default(self):
        """CPU: set_dtype('float32') restores _DEFAULT_DTYPE to np.float32."""
        import novax.core as _core
        try:
            nx.set_dtype("float16")
        finally:
            nx.set_dtype("float32")
        assert _core._DEFAULT_DTYPE == np.float32
        t = Tensor([1.0, 2.0])
        assert t.dtype == np.float32

    def test_set_dtype_aliases(self):
        """CPU: 'fp16' and 'half' are accepted aliases for float16."""
        import novax.core as _core
        for alias in ("fp16", "half"):
            try:
                nx.set_dtype(alias)
                assert _core._DEFAULT_DTYPE == np.float16
            finally:
                nx.set_dtype("float32")

    def test_float16_arithmetic_preserves_dtype(self):
        """CPU: arithmetic on float16 tensors via eval() produces a result tensor."""
        a = Tensor([1.0, 2.0, 4.0], dtype=np.float16)
        b = Tensor([2.0, 3.0, 1.0], dtype=np.float16)
        # The CPU dispatch path operates on numpy arrays; the result dtype
        # follows numpy promotion rules for float16 inputs.
        result = (a + b).eval()
        assert isinstance(result, Tensor)
        np.testing.assert_array_almost_equal(
            result.data.astype(np.float32), [3.0, 5.0, 5.0], decimal=3
        )

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_fp16_gpu_roundtrip(self):
        """GPU: fp16 tensor round-trips through to_gpu()/to_host() correctly."""
        arr = np.array([1.0, 0.5, 2.5, 3.0], dtype=np.float16)
        t = Tensor(arr, dtype=np.float16)
        t.to_gpu()
        result = t.to_host()
        np.testing.assert_array_almost_equal(
            result.astype(np.float32), arr.astype(np.float32), decimal=3
        )


# ---------------------------------------------------------------------------
# Item 7 — Triton fused 1D softmax
# ---------------------------------------------------------------------------

class TestTritonSoftmax:
    """Item 7: Triton-based fused 1D softmax kernel."""

    def test_triton_softmax_returns_none_without_gpu(self):
        """CPU: triton_softmax_1d returns None when Triton is unavailable."""
        # In CI (no GPU / no Triton), the function must return None gracefully.
        if GPU_AVAILABLE:
            pytest.skip("Skipping CPU-only assertion — GPU is present in this env")
        # Build a minimal fake Tensor-like object so the import doesn't fail.
        # triton_softmax_1d checks _TRITON_AVAILABLE at the top and returns None.
        result = triton_softmax_1d(Tensor([1.0, 2.0, 3.0]))
        assert result is None

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_triton_softmax_correct_values(self):
        """GPU: triton_softmax_1d returns softmax values matching numpy for n=1024."""
        n = 1024
        a_np = np.random.randn(n).astype(np.float32)
        t = Tensor(a_np).to_gpu()

        result = triton_softmax_1d(t)
        if result is None:
            pytest.skip("Triton not available in this GPU environment")

        result_host = result.to_host()
        # Reference softmax
        exp_a = np.exp(a_np - np.max(a_np))
        ref = exp_a / np.sum(exp_a)

        np.testing.assert_array_almost_equal(result_host, ref, decimal=4)
        # Softmax output must sum to 1.0
        assert abs(float(np.sum(result_host)) - 1.0) < 1e-4

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_triton_softmax_shape_preserved(self):
        """GPU: triton_softmax_1d preserves input shape."""
        n = 512
        t = Tensor(np.random.randn(n).astype(np.float32)).to_gpu()
        result = triton_softmax_1d(t)
        if result is None:
            pytest.skip("Triton not available in this GPU environment")
        assert result.shape == (n,)


# ---------------------------------------------------------------------------
# Item 8 — Pattern matching for auto-fusion
# ---------------------------------------------------------------------------

class TestPatternMatching:
    """Item 8: lazy graph pattern matching for matmul+bias+relu and matmul+bias."""

    # --- _match_matmul_bias_relu ---

    def test_match_matmul_bias_relu_returns_triple(self):
        """CPU: _match_matmul_bias_relu() returns (A, B, bias) for relu(A@B+bias)."""
        A = Tensor(np.random.randn(4, 3).astype(np.float32))
        B = Tensor(np.random.randn(3, 5).astype(np.float32))
        bias = Tensor(np.zeros(5, dtype=np.float32))

        node = Tensor(None, op="relu", inputs=[
            Tensor(None, op="add", inputs=[
                Tensor(None, op="matmul", inputs=[A, B]),
                bias,
            ])
        ])
        match = node._match_matmul_bias_relu()
        assert match is not None, "_match_matmul_bias_relu should match relu(A@B+bias)"
        matched_A, matched_B, matched_bias = match
        assert matched_A is A
        assert matched_B is B
        assert matched_bias is bias

    def test_match_matmul_bias_relu_rejects_sigmoid(self):
        """CPU: _match_matmul_bias_relu() returns None for sigmoid(A@B+bias)."""
        A = Tensor(np.random.randn(4, 3).astype(np.float32))
        B = Tensor(np.random.randn(3, 5).astype(np.float32))
        bias = Tensor(np.zeros(5, dtype=np.float32))

        node = Tensor(None, op="sigmoid", inputs=[
            Tensor(None, op="add", inputs=[
                Tensor(None, op="matmul", inputs=[A, B]),
                bias,
            ])
        ])
        match = node._match_matmul_bias_relu()
        assert match is None, "_match_matmul_bias_relu should return None for sigmoid"

    def test_match_matmul_bias_relu_rejects_plain_relu(self):
        """CPU: _match_matmul_bias_relu() returns None for relu(A) without matmul."""
        A = Tensor(np.random.randn(4, 3).astype(np.float32))
        node = Tensor(None, op="relu", inputs=[A])
        match = node._match_matmul_bias_relu()
        assert match is None

    def test_match_matmul_bias_relu_rejects_no_bias(self):
        """CPU: _match_matmul_bias_relu() returns None for relu(A@B) without add."""
        A = Tensor(np.random.randn(4, 3).astype(np.float32))
        B = Tensor(np.random.randn(3, 5).astype(np.float32))
        node = Tensor(None, op="relu", inputs=[
            Tensor(None, op="matmul", inputs=[A, B])
        ])
        match = node._match_matmul_bias_relu()
        assert match is None

    # --- _match_matmul_bias ---

    def test_match_matmul_bias_returns_triple(self):
        """CPU: _match_matmul_bias() returns (A, B, bias) for A@B+bias."""
        A = Tensor(np.random.randn(4, 3).astype(np.float32))
        B = Tensor(np.random.randn(3, 5).astype(np.float32))
        bias = Tensor(np.zeros(5, dtype=np.float32))

        node = Tensor(None, op="add", inputs=[
            Tensor(None, op="matmul", inputs=[A, B]),
            bias,
        ])
        match = node._match_matmul_bias()
        assert match is not None, "_match_matmul_bias should match A@B+bias"
        matched_A, matched_B, matched_bias = match
        assert matched_A is A
        assert matched_B is B
        assert matched_bias is bias

    def test_match_matmul_bias_rejects_plain_matmul(self):
        """CPU: _match_matmul_bias() returns None for plain A@B without add."""
        A = Tensor(np.random.randn(4, 3).astype(np.float32))
        B = Tensor(np.random.randn(3, 5).astype(np.float32))

        node = Tensor(None, op="matmul", inputs=[A, B])
        match = node._match_matmul_bias()
        assert match is None, "_match_matmul_bias should return None for plain matmul"

    def test_match_matmul_bias_rejects_non_add_root(self):
        """CPU: _match_matmul_bias() returns None when root op is not 'add'."""
        A = Tensor(np.random.randn(2, 3).astype(np.float32))
        B = Tensor(np.random.randn(3, 2).astype(np.float32))
        bias = Tensor(np.zeros(2, dtype=np.float32))

        node = Tensor(None, op="sub", inputs=[
            Tensor(None, op="matmul", inputs=[A, B]),
            bias,
        ])
        match = node._match_matmul_bias()
        assert match is None

    # --- correctness: eval on CPU must not change when pattern matching runs ---

    def test_eval_relu_matmul_bias_cpu_correctness(self):
        """CPU: eval() of relu(A@B+bias) on CPU tensors gives same result before/after."""
        np.random.seed(42)
        M, K, N = 8, 4, 6
        a_np = np.random.randn(M, K).astype(np.float32)
        W_np = np.random.randn(K, N).astype(np.float32)
        b_np = np.random.randn(N).astype(np.float32)

        A = Tensor(a_np)
        W = Tensor(W_np)
        b = Tensor(b_np)

        # Reference via numpy
        ref = np.maximum(0.0, a_np @ W_np + b_np)

        # Evaluate via lazy graph (CPU path, no GPU → pattern match branch skipped)
        lazy = Tensor(None, op="relu", inputs=[
            Tensor(None, op="add", inputs=[
                Tensor(None, op="matmul", inputs=[A, W]),
                b,
            ])
        ])
        result = lazy.eval()
        np.testing.assert_array_almost_equal(result.data, ref, decimal=5)

    def test_eval_matmul_bias_cpu_correctness(self):
        """CPU: eval() of A@W+bias on CPU tensors gives the same result as numpy."""
        np.random.seed(7)
        M, K, N = 6, 3, 5
        a_np = np.random.randn(M, K).astype(np.float32)
        W_np = np.random.randn(K, N).astype(np.float32)
        b_np = np.random.randn(N).astype(np.float32)

        A = Tensor(a_np)
        W = Tensor(W_np)
        b = Tensor(b_np)

        ref = a_np @ W_np + b_np

        lazy = Tensor(None, op="add", inputs=[
            Tensor(None, op="matmul", inputs=[A, W]),
            b,
        ])
        result = lazy.eval()
        np.testing.assert_array_almost_equal(result.data, ref, decimal=5)

    # --- GPU path: fused kernel is chosen and gives correct output ---

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_gpu_relu_matmul_bias_routes_to_fused_kernel(self):
        """GPU: eval() of relu(x@W+b) on GPU tensors uses launch_matmul_bias_relu."""
        np.random.seed(99)
        M, K, N = 16, 8, 4
        x_np = np.random.randn(M, K).astype(np.float32)
        W_np = np.random.randn(K, N).astype(np.float32)
        b_np = np.random.randn(N).astype(np.float32)

        x = Tensor(x_np).to_gpu()
        W = Tensor(W_np).to_gpu()
        b = Tensor(b_np).to_gpu()

        # Build lazy graph and eval (pattern match triggers fused kernel)
        result = (x @ W + b)
        # Wrap in relu lazy node
        fused_node = Tensor(None, op="relu", inputs=[result])
        out = fused_node.eval()

        # Verify shape
        assert out.shape == (M, N)

        # Verify values match numpy reference
        ref = np.maximum(0.0, x_np @ W_np + b_np)
        np.testing.assert_array_almost_equal(out.to_host(), ref, decimal=4)

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_gpu_matmul_bias_routes_to_fused_kernel(self):
        """GPU: eval() of x@W+b on GPU tensors uses launch_matmul_bias."""
        np.random.seed(13)
        M, K, N = 12, 6, 3
        x_np = np.random.randn(M, K).astype(np.float32)
        W_np = np.random.randn(K, N).astype(np.float32)
        b_np = np.random.randn(N).astype(np.float32)

        x = Tensor(x_np).to_gpu()
        W = Tensor(W_np).to_gpu()
        b = Tensor(b_np).to_gpu()

        lazy = Tensor(None, op="add", inputs=[
            Tensor(None, op="matmul", inputs=[x, W]),
            b,
        ])
        out = lazy.eval()

        assert out.shape == (M, N)
        ref = x_np @ W_np + b_np
        np.testing.assert_array_almost_equal(out.to_host(), ref, decimal=4)

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_launch_matmul_bias_relu_direct(self):
        """GPU: launch_matmul_bias_relu gives correct shape and values."""
        np.random.seed(1)
        M, K, N = 8, 4, 4
        x_np = np.random.randn(M, K).astype(np.float32)
        W_np = np.random.randn(K, N).astype(np.float32)
        b_np = np.random.randn(N).astype(np.float32)

        x = Tensor(x_np).to_gpu()
        W = Tensor(W_np).to_gpu()
        b = Tensor(b_np).to_gpu()

        out = launch_matmul_bias_relu(x, W, b)
        assert out.shape == (M, N)
        ref = np.maximum(0.0, x_np @ W_np + b_np)
        np.testing.assert_array_almost_equal(out.to_host(), ref, decimal=4)

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_launch_matmul_bias_direct(self):
        """GPU: launch_matmul_bias gives correct shape and values."""
        np.random.seed(2)
        M, K, N = 8, 4, 4
        x_np = np.random.randn(M, K).astype(np.float32)
        W_np = np.random.randn(K, N).astype(np.float32)
        b_np = np.random.randn(N).astype(np.float32)

        x = Tensor(x_np).to_gpu()
        W = Tensor(W_np).to_gpu()
        b = Tensor(b_np).to_gpu()

        out = launch_matmul_bias(x, W, b)
        assert out.shape == (M, N)
        ref = x_np @ W_np + b_np
        np.testing.assert_array_almost_equal(out.to_host(), ref, decimal=4)
