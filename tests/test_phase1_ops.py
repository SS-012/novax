import pytest
import numpy as np
import novax as nx
from novax.core import Tensor, GPU_AVAILABLE


# ---------------------------------------------------------------------------
# Unary elementwise ops
# ---------------------------------------------------------------------------

class TestUnaryOps:
    def test_exp(self):
        a = Tensor(np.array([0.0, 1.0, 2.0]))
        result = nx.exp(a)
        assert isinstance(result, Tensor)
        np.testing.assert_array_almost_equal(result.data, np.exp([0.0, 1.0, 2.0]))

    def test_log(self):
        a = Tensor(np.array([1.0, np.e, np.e ** 2]))
        result = nx.log(a)
        np.testing.assert_array_almost_equal(result.data, [0.0, 1.0, 2.0], decimal=5)

    def test_log_nonpositive_raises(self):
        a = Tensor(np.array([0.0, 1.0]))
        with pytest.raises(ValueError):
            nx.log(a)

    def test_sqrt(self):
        a = Tensor(np.array([0.0, 1.0, 4.0, 9.0]))
        result = nx.sqrt(a)
        np.testing.assert_array_almost_equal(result.data, [0.0, 1.0, 2.0, 3.0])

    def test_sqrt_negative_raises(self):
        a = Tensor(np.array([-1.0]))
        with pytest.raises(ValueError):
            nx.sqrt(a)

    def test_abs(self):
        a = Tensor(np.array([-3.0, 0.0, 4.0]))
        result = nx.abs(a)
        np.testing.assert_array_almost_equal(result.data, [3.0, 0.0, 4.0])

    def test_neg(self):
        a = Tensor(np.array([1.0, -2.0, 3.0]))
        result = nx.neg(a)
        np.testing.assert_array_almost_equal(result.data, [-1.0, 2.0, -3.0])

    def test_neg_operator(self):
        a = Tensor(np.array([1.0, 2.0]))
        result = (-a).eval()
        np.testing.assert_array_almost_equal(result.data, [-1.0, -2.0])

    def test_result_is_tensor(self):
        a = Tensor(np.array([1.0, 2.0]))
        for fn in [nx.exp, nx.sqrt, nx.abs, nx.neg]:
            assert isinstance(fn(a), Tensor)


# ---------------------------------------------------------------------------
# Pow (binary)
# ---------------------------------------------------------------------------

class TestPow:
    def test_pow_elementwise(self):
        a = Tensor(np.array([2.0, 3.0, 4.0]))
        b = Tensor(np.array([3.0, 2.0, 0.5]))
        result = nx.pow(a, b)
        np.testing.assert_array_almost_equal(result.data, [8.0, 9.0, 2.0])

    def test_pow_operator(self):
        a = Tensor(np.array([2.0, 3.0]))
        b = Tensor(np.array([2.0, 2.0]))
        result = (a ** b).eval()
        np.testing.assert_array_almost_equal(result.data, [4.0, 9.0])


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

class TestActivations:
    def test_relu_positive(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]))
        result = nx.relu(a)
        np.testing.assert_array_almost_equal(result.data, [1.0, 2.0, 3.0])

    def test_relu_negative(self):
        a = Tensor(np.array([-1.0, -2.0, 0.0]))
        result = nx.relu(a)
        np.testing.assert_array_almost_equal(result.data, [0.0, 0.0, 0.0])

    def test_relu_mixed(self):
        a = Tensor(np.array([-2.0, 0.0, 3.0, -0.5, 5.0]))
        result = nx.relu(a)
        np.testing.assert_array_almost_equal(result.data, [0.0, 0.0, 3.0, 0.0, 5.0])

    def test_sigmoid_range(self):
        a = Tensor(np.array([-100.0, 0.0, 100.0]))
        result = nx.sigmoid(a)
        assert float(result.data[0]) < 0.001
        assert abs(float(result.data[1]) - 0.5) < 1e-4
        assert float(result.data[2]) > 0.999

    def test_tanh_range(self):
        a = Tensor(np.array([-100.0, 0.0, 100.0]))
        result = nx.tanh(a)
        assert float(result.data[0]) < -0.999
        assert abs(float(result.data[1])) < 1e-4
        assert float(result.data[2]) > 0.999

    def test_softmax_sums_to_one(self):
        a = Tensor(np.array([1.0, 2.0, 3.0, 4.0]))
        result = nx.softmax(a)
        assert abs(float(np.sum(result.data)) - 1.0) < 1e-5

    def test_softmax_monotone(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]))
        result = nx.softmax(a)
        assert result.data[0] < result.data[1] < result.data[2]

    def test_softmax_numerical_stability(self):
        a = Tensor(np.array([1000.0, 1000.0, 1000.0]))
        result = nx.softmax(a)
        np.testing.assert_array_almost_equal(result.data, [1/3, 1/3, 1/3], decimal=5)


# ---------------------------------------------------------------------------
# Reduction ops
# ---------------------------------------------------------------------------

class TestReductions:
    def test_sum(self):
        a = Tensor(np.array([1.0, 2.0, 3.0, 4.0]))
        result = nx.sum(a)
        assert isinstance(result, Tensor)
        assert abs(float(result.data[0]) - 10.0) < 1e-5

    def test_mean(self):
        a = Tensor(np.array([1.0, 2.0, 3.0, 4.0]))
        result = nx.mean(a)
        assert abs(float(result.data[0]) - 2.5) < 1e-5

    def test_max(self):
        a = Tensor(np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]))
        result = nx.max(a)
        assert abs(float(result.data[0]) - 9.0) < 1e-5

    def test_min(self):
        a = Tensor(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))
        result = nx.min(a)
        assert abs(float(result.data[0]) - 1.0) < 1e-5

    def test_sum_single_element(self):
        a = Tensor(np.array([7.0]))
        result = nx.sum(a)
        assert abs(float(result.data[0]) - 7.0) < 1e-5

    def test_reduction_returns_tensor(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]))
        for fn in [nx.sum, nx.mean, nx.max, nx.min]:
            assert isinstance(fn(a), Tensor)


# ---------------------------------------------------------------------------
# Matrix multiplication
# ---------------------------------------------------------------------------

class TestMatmul:
    def test_basic_matmul(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
        result = nx.matmul(a, b)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert isinstance(result, Tensor)
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_matmul_shape(self):
        a = Tensor(np.random.randn(3, 4).astype(np.float32))
        b = Tensor(np.random.randn(4, 5).astype(np.float32))
        result = nx.matmul(a, b)
        assert result.shape == (3, 5)

    def test_matmul_operator(self):
        a = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]))
        b = Tensor(np.array([[3.0, 4.0], [5.0, 6.0]]))
        result = (a @ b).eval()
        np.testing.assert_array_almost_equal(result.data, [[3.0, 4.0], [5.0, 6.0]])

    def test_matmul_non_square(self):
        a = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))  # (2,3)
        b = Tensor(np.array([[7.0], [8.0], [9.0]]))                 # (3,1)
        result = nx.matmul(a, b)
        expected = np.array([[50.0], [122.0]])
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_matmul_shape_mismatch_raises(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))  # (2,2)
        b = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))  # (3,2)
        with pytest.raises((ValueError, AssertionError)):
            nx.matmul(a, b)

    def test_matmul_vs_numpy(self):
        np.random.seed(42)
        a_np = np.random.randn(10, 8).astype(np.float32)
        b_np = np.random.randn(8, 6).astype(np.float32)
        a = Tensor(a_np)
        b = Tensor(b_np)
        result = nx.matmul(a, b)
        np.testing.assert_array_almost_equal(result.data, np.matmul(a_np, b_np), decimal=4)


# ---------------------------------------------------------------------------
# Reshape and transpose
# ---------------------------------------------------------------------------

class TestReshapeTranspose:
    def test_reshape(self):
        a = Tensor(np.arange(6, dtype=np.float32))
        b = a.reshape(2, 3)
        assert b.shape == (2, 3)
        np.testing.assert_array_equal(b.data, [[0, 1, 2], [3, 4, 5]])

    def test_transpose_2d(self):
        a = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        b = a.transpose()
        assert b.shape == (3, 2)
        np.testing.assert_array_equal(b.data, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    def test_reshape_invalid_size_raises(self):
        a = Tensor(np.arange(6, dtype=np.float32))
        with pytest.raises((ValueError, Exception)):
            a.reshape(2, 4)


# ---------------------------------------------------------------------------
# Eval with new ops (lazy graph)
# ---------------------------------------------------------------------------

class TestEvalNewOps:
    def test_eval_exp(self):
        a = Tensor(np.array([0.0, 1.0]))
        node = Tensor(None, op="exp", inputs=[a])
        result = node.eval()
        np.testing.assert_array_almost_equal(result.data, np.exp([0.0, 1.0]))

    def test_eval_relu(self):
        a = Tensor(np.array([-1.0, 0.0, 2.0]))
        node = Tensor(None, op="relu", inputs=[a])
        result = node.eval()
        np.testing.assert_array_almost_equal(result.data, [0.0, 0.0, 2.0])

    def test_eval_sum(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]))
        node = Tensor(None, op="sum", inputs=[a])
        result = node.eval()
        assert abs(float(result.data[0]) - 6.0) < 1e-5

    def test_eval_matmul(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]))
        node = Tensor(None, op="matmul", inputs=[a, b])
        result = node.eval()
        np.testing.assert_array_almost_equal(result.data, a.data)

    def test_eval_fused_relu_add(self):
        a = Tensor(np.array([-1.0, 2.0, -3.0]))
        b = Tensor(np.array([2.0, 0.0, 4.0]))
        node = Tensor(None, op="relu", inputs=[a + b])
        result = node.eval()
        np.testing.assert_array_almost_equal(result.data, [1.0, 2.0, 1.0])


# ---------------------------------------------------------------------------
# no_grad context manager
# ---------------------------------------------------------------------------

class TestNoGrad:
    def test_no_grad_disables_tracking(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        with nx.no_grad():
            c = (a + b).eval()
        assert not c.requires_grad

    def test_grad_reenabled_after_context(self):
        from novax.autograd import _get_grad_enabled
        with nx.no_grad():
            assert not _get_grad_enabled()
        assert _get_grad_enabled()

    def test_grad_enabled_by_default(self):
        from novax.autograd import _get_grad_enabled
        assert _get_grad_enabled()


# ---------------------------------------------------------------------------
# GPU new ops (skipped without GPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestGPUNewOps:
    def test_gpu_relu(self):
        a = Tensor(np.array([-1.0, 0.0, 2.0])).to_gpu()
        nx.set_default_device("gpu")
        result = nx.relu(a)
        np.testing.assert_array_almost_equal(result.to_host(), [0.0, 0.0, 2.0])

    def test_gpu_exp(self):
        a = Tensor(np.array([0.0, 1.0])).to_gpu()
        nx.set_default_device("gpu")
        result = nx.exp(a)
        np.testing.assert_array_almost_equal(result.to_host(), np.exp([0.0, 1.0]))

    def test_gpu_abs_materializes_on_to_host(self):
        a = Tensor(np.array([-2.0, 0.0, 3.0], dtype=np.float32)).to_gpu()
        nx.set_default_device("gpu")
        result = nx.abs(a)
        np.testing.assert_array_almost_equal(result.to_host(), [2.0, 0.0, 3.0])

    def test_gpu_matmul(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]])).to_gpu()
        b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]])).to_gpu()
        nx.set_default_device("gpu")
        result = nx.matmul(a, b)
        np.testing.assert_array_almost_equal(result.to_host(), [[19.0, 22.0], [43.0, 50.0]])

    def test_gpu_matmul_medium(self):
        rng = np.random.default_rng(123)
        a_arr = rng.standard_normal((128, 128), dtype=np.float32)
        b_arr = rng.standard_normal((128, 128), dtype=np.float32)
        a = Tensor(a_arr).to_gpu()
        b = Tensor(b_arr).to_gpu()
        nx.set_default_device("gpu")
        result = nx.matmul(a, b)
        np.testing.assert_array_almost_equal(result.to_host(), a_arr @ b_arr, decimal=3)

    def test_gpu_sum(self):
        a = Tensor(np.array([1.0, 2.0, 3.0, 4.0])).to_gpu()
        nx.set_default_device("gpu")
        result = nx.sum(a)
        assert abs(float(result.to_host()[0]) - 10.0) < 1e-4

    def test_gpu_large_sum_and_mean(self):
        arr = np.linspace(-1.0, 1.0, 65536, dtype=np.float32)
        a = Tensor(arr).to_gpu()
        nx.set_default_device("gpu")
        sum_result = nx.sum(a)
        mean_result = nx.mean(a)
        assert abs(float(sum_result.to_host()[0]) - float(np.sum(arr))) < 1e-2
        assert abs(float(mean_result.to_host()[0]) - float(np.mean(arr))) < 1e-5

    def test_gpu_softmax(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a = Tensor(arr).to_gpu()
        nx.set_default_device("gpu")
        result = nx.softmax(a).to_host()
        expected = np.exp(arr - np.max(arr))
        expected = expected / np.sum(expected)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_gpu_lazy_bias_broadcast(self):
        a_arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        b_arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        a = Tensor(a_arr).to_gpu()
        b = Tensor(b_arr).to_gpu()
        nx.set_default_device("gpu")
        result = (a + b).eval()
        np.testing.assert_array_almost_equal(result.to_host(), a_arr + b_arr)

    def test_gpu_mlp_forward_bias_broadcast(self):
        x_arr = np.arange(8, dtype=np.float32).reshape(2, 4) / 8.0
        w_arr = np.arange(12, dtype=np.float32).reshape(4, 3) / 12.0
        b_arr = np.array([0.25, -0.5, 0.75], dtype=np.float32)
        x = Tensor(x_arr).to_gpu()
        w = Tensor(w_arr).to_gpu()
        b = Tensor(b_arr).to_gpu()
        nx.set_default_device("gpu")
        result = nx.relu(nx.matmul(x, w) + b).eval()
        np.testing.assert_array_almost_equal(result.to_host(), np.maximum(x_arr @ w_arr + b_arr, 0.0), decimal=5)

    def test_gpu_lazy_nested_elementwise_chain(self):
        a_arr = np.linspace(-2.0, 2.0, 64, dtype=np.float32)
        b_arr = a_arr * 0.5
        c_arr = a_arr + 1.0
        a = Tensor(a_arr).to_gpu()
        b = Tensor(b_arr).to_gpu()
        c = Tensor(c_arr).to_gpu()
        nx.set_default_device("gpu")
        result = nx.sigmoid(nx.relu(a * b + c) * a).eval()
        expected = 1.0 / (1.0 + np.exp(-(np.maximum(a_arr * b_arr + c_arr, 0.0) * a_arr)))
        np.testing.assert_array_almost_equal(result.to_host(), expected, decimal=5)

    def test_gpu_lazy_sqrt_abs_chain(self):
        a_arr = np.linspace(-4.0, 4.0, 64, dtype=np.float32)
        a = Tensor(a_arr).to_gpu()
        nx.set_default_device("gpu")
        result = nx.sqrt(nx.abs(a)).eval()
        np.testing.assert_array_almost_equal(result.to_host(), np.sqrt(np.abs(a_arr)), decimal=5)

    def test_cuda_graph_capture_replay(self):
        a = Tensor(np.arange(16, dtype=np.float32)).to_gpu()
        nx.set_default_device("gpu")
        graph = nx.CUDAGraph()
        graph.capture(lambda: nx.neg(a).eval())
        graph.replay()
        graph.replay_many(2)
