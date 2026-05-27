import pytest
import numpy as np
import novax as nx
from novax.core import Tensor, GPU_AVAILABLE


def numerical_grad(fn, tensor, eps=1e-3):
    """Compute numerical gradient via finite differences."""
    arr = tensor.data.copy()
    grad = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = arr[idx]

        arr[idx] = orig + eps
        tensor.data = arr.copy()
        fplus = float(fn().data.flat[0])

        arr[idx] = orig - eps
        tensor.data = arr.copy()
        fminus = float(fn().data.flat[0])

        grad[idx] = (fplus - fminus) / (2 * eps)
        arr[idx] = orig
        it.iternext()

    tensor.data = arr
    return grad


# ---------------------------------------------------------------------------
# Basic gradient properties
# ---------------------------------------------------------------------------

class TestGradBasics:
    def test_leaf_has_grad_none_initially(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        assert a.grad is None

    def test_backward_sets_grad(self):
        a = Tensor(np.array([2.0]), requires_grad=True)
        b = Tensor(np.array([3.0]))
        c = (a + b).eval()
        nx.sum(c).eval().backward()
        assert a.grad is not None

    def test_no_grad_leaf_not_tracked(self):
        a = Tensor(np.array([1.0]))  # requires_grad=False by default
        b = Tensor(np.array([2.0]), requires_grad=True)
        c = (a + b).eval()
        nx.sum(c).eval().backward()
        assert a.grad is None
        assert b.grad is not None

    def test_requires_grad_propagates(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        b = Tensor(np.array([3.0, 4.0]))
        c = (a + b).eval()
        assert c.requires_grad is True

    def test_requires_grad_false_both_no_tracking(self):
        a = Tensor(np.array([1.0]))
        b = Tensor(np.array([2.0]))
        c = (a + b).eval()
        assert c.requires_grad is False


# ---------------------------------------------------------------------------
# Add / Sub gradients
# ---------------------------------------------------------------------------

class TestAddSubGrad:
    def test_add_gradient_both(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        b = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)
        c = (a + b).eval()
        nx.sum(c).eval().backward()
        np.testing.assert_array_almost_equal(a.grad.data, [1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(b.grad.data, [1.0, 1.0, 1.0])

    def test_sub_gradient(self):
        a = Tensor(np.array([5.0, 6.0]), requires_grad=True)
        b = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        c = (a - b).eval()
        nx.sum(c).eval().backward()
        np.testing.assert_array_almost_equal(a.grad.data, [1.0, 1.0])
        np.testing.assert_array_almost_equal(b.grad.data, [-1.0, -1.0])


# ---------------------------------------------------------------------------
# Mul / Div gradients
# ---------------------------------------------------------------------------

class TestMulDivGrad:
    def test_mul_gradient(self):
        a_arr = np.array([2.0, 3.0])
        b_arr = np.array([4.0, 5.0])
        a = Tensor(a_arr.copy(), requires_grad=True)
        b = Tensor(b_arr.copy(), requires_grad=True)
        c = (a * b).eval()
        nx.sum(c).eval().backward()
        np.testing.assert_array_almost_equal(a.grad.data, b_arr)
        np.testing.assert_array_almost_equal(b.grad.data, a_arr)

    def test_div_gradient_numerically(self):
        a_arr = np.array([6.0, 8.0])
        b_arr = np.array([2.0, 4.0])
        a = Tensor(a_arr.copy(), requires_grad=True)
        b = Tensor(b_arr.copy(), requires_grad=True)
        c = (a / b).eval()
        nx.sum(c).eval().backward()
        # ∂(a/b)/∂a = 1/b
        np.testing.assert_array_almost_equal(a.grad.data, 1.0 / b_arr, decimal=5)
        # ∂(a/b)/∂b = -a/b²
        np.testing.assert_array_almost_equal(b.grad.data, -a_arr / (b_arr ** 2), decimal=5)


# ---------------------------------------------------------------------------
# Unary gradients (numerical check)
# ---------------------------------------------------------------------------

class TestUnaryGrad:
    def _check_grad(self, fn, a_arr, decimal=3):
        a = Tensor(a_arr.copy(), requires_grad=True)

        def forward():
            a2 = Tensor(a.data.copy(), requires_grad=True)
            return nx.sum(fn(a2)).eval()

        num_g = numerical_grad(forward, Tensor(a_arr.copy()), eps=1e-3)

        a = Tensor(a_arr.copy(), requires_grad=True)
        nx.sum(fn(a)).eval().backward()
        np.testing.assert_array_almost_equal(a.grad.data, num_g, decimal=decimal)

    def test_exp_grad(self):
        a = Tensor(np.array([0.5, 1.0, 1.5]), requires_grad=True)
        nx.sum(nx.exp(a)).eval().backward()
        np.testing.assert_array_almost_equal(a.grad.data, np.exp([0.5, 1.0, 1.5]), decimal=4)

    def test_log_grad(self):
        a = Tensor(np.array([1.0, 2.0, 4.0]), requires_grad=True)
        nx.sum(nx.log(a)).eval().backward()
        np.testing.assert_array_almost_equal(a.grad.data, [1.0, 0.5, 0.25], decimal=5)

    def test_relu_grad_positive(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        nx.sum(nx.relu(a)).eval().backward()
        np.testing.assert_array_almost_equal(a.grad.data, [1.0, 1.0, 1.0])

    def test_relu_grad_negative(self):
        a = Tensor(np.array([-1.0, -2.0]), requires_grad=True)
        nx.sum(nx.relu(a)).eval().backward()
        np.testing.assert_array_almost_equal(a.grad.data, [0.0, 0.0])

    def test_neg_grad(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        nx.sum(nx.neg(a)).eval().backward()
        np.testing.assert_array_almost_equal(a.grad.data, [-1.0, -1.0])


# ---------------------------------------------------------------------------
# Reduction gradients
# ---------------------------------------------------------------------------

class TestReductionGrad:
    def test_sum_grad(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        nx.sum(a).eval().backward()
        np.testing.assert_array_almost_equal(a.grad.data, [1.0, 1.0, 1.0])

    def test_mean_grad(self):
        a = Tensor(np.array([1.0, 2.0, 3.0, 4.0]), requires_grad=True)
        nx.mean(a).eval().backward()
        np.testing.assert_array_almost_equal(a.grad.data, [0.25, 0.25, 0.25, 0.25])


# ---------------------------------------------------------------------------
# Matmul gradients
# ---------------------------------------------------------------------------

class TestMatmulGrad:
    def test_matmul_grad_left(self):
        A_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        B_arr = np.array([[5.0, 6.0], [7.0, 8.0]])
        A = Tensor(A_arr.copy(), requires_grad=True)
        B = Tensor(B_arr.copy())
        out = nx.matmul(A, B)
        nx.sum(out).eval().backward()
        # ∂/∂A = grad_out @ B.T; grad_out = ones(2,2)
        expected = np.ones((2, 2)) @ B_arr.T
        np.testing.assert_array_almost_equal(A.grad.data, expected)

    def test_matmul_grad_right(self):
        A_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        B_arr = np.array([[5.0, 6.0], [7.0, 8.0]])
        A = Tensor(A_arr.copy())
        B = Tensor(B_arr.copy(), requires_grad=True)
        out = nx.matmul(A, B)
        nx.sum(out).eval().backward()
        # ∂/∂B = A.T @ grad_out; grad_out = ones(2,2)
        expected = A_arr.T @ np.ones((2, 2))
        np.testing.assert_array_almost_equal(B.grad.data, expected)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestGPUAutograd:
    def _gpu_tensor(self, arr, requires_grad=False):
        return Tensor(np.array(arr, dtype=np.float32), requires_grad=requires_grad).to_gpu()

    def test_gpu_mean_backward_stays_on_device(self):
        a_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        a = self._gpu_tensor(a_arr, requires_grad=True)

        nx.mean(a).eval().backward()

        assert a.grad is not None
        assert a.grad.on_gpu is True
        np.testing.assert_allclose(a.grad.to_host(), np.full_like(a_arr, 0.25))

    def test_gpu_relu_sum_backward_stays_on_device(self):
        a_arr = np.array([-2.0, 0.0, 3.0, 4.0], dtype=np.float32)
        a = self._gpu_tensor(a_arr, requires_grad=True)

        nx.sum(nx.relu(a)).eval().backward()

        assert a.grad is not None
        assert a.grad.on_gpu is True
        np.testing.assert_allclose(a.grad.to_host(), [0.0, 0.0, 1.0, 1.0])

    def test_gpu_matmul_bias_mean_backward(self):
        x_arr = (np.arange(12, dtype=np.float32).reshape(3, 4) / 10.0)
        w_arr = (np.arange(8, dtype=np.float32).reshape(4, 2) / 10.0)
        b_arr = np.zeros(2, dtype=np.float32)
        x = self._gpu_tensor(x_arr)
        w = self._gpu_tensor(w_arr, requires_grad=True)
        b = self._gpu_tensor(b_arr, requires_grad=True)

        nx.mean(nx.matmul(x, w) + b).eval().backward()

        grad_out = np.ones((3, 2), dtype=np.float32) / 6.0
        assert w.grad is not None
        assert b.grad is not None
        assert w.grad.on_gpu is True
        assert b.grad.on_gpu is True
        np.testing.assert_allclose(w.grad.to_host(), x_arr.T @ grad_out, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(b.grad.to_host(), grad_out.sum(axis=0), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Chain rule (multi-op backward)
# ---------------------------------------------------------------------------

class TestChainRule:
    def test_chain_add_mul(self):
        a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        b = Tensor(np.array([4.0, 5.0]))
        c = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        # loss = sum((a + b) * c)
        out = nx.sum((a + b) * c).eval()
        out.backward()
        # ∂loss/∂a = c = [1, 2]
        np.testing.assert_array_almost_equal(a.grad.data, [1.0, 2.0])
        # ∂loss/∂c = (a + b) = [6, 8]
        np.testing.assert_array_almost_equal(c.grad.data, [6.0, 8.0])

    def test_mlp_forward_backward(self):
        np.random.seed(0)
        x = Tensor(np.random.randn(4, 8).astype(np.float32))
        W = Tensor(np.random.randn(8, 4).astype(np.float32), requires_grad=True)
        b = Tensor(np.zeros(4, dtype=np.float32), requires_grad=True)

        hidden = nx.relu(nx.matmul(x, W) + b)
        loss = nx.mean(hidden)
        loss.eval().backward()

        assert W.grad is not None
        assert b.grad is not None
        assert W.grad.shape == (8, 4)
        assert b.grad.shape == (4,)


# ---------------------------------------------------------------------------
# no_grad prevents gradient accumulation
# ---------------------------------------------------------------------------

class TestNoGradAutograd:
    def test_no_grad_no_backward_closure(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        with nx.no_grad():
            c = (a + b).eval()
        assert not c.requires_grad
        assert len(c._prev) == 0

    def test_grad_reenabled_after_no_grad(self):
        a = Tensor(np.array([1.0]), requires_grad=True)
        b = Tensor(np.array([2.0]), requires_grad=True)
        with nx.no_grad():
            _ = (a + b).eval()
        c = (a + b).eval()
        assert c.requires_grad
