import pytest
import numpy as np
import novax as nx
from novax.core import Tensor, GPU_AVAILABLE
from novax.utils import mempool


# ---------------------------------------------------------------------------
# Tensor construction
# ---------------------------------------------------------------------------

class TestTensorCreation:
    def test_from_list(self):
        t = Tensor([1, 2, 3])
        assert isinstance(t.data, np.ndarray)
        assert t.shape == (3,)
        assert t.dtype == np.float32
        assert t.size == 3
        assert t.is_leaf is True
        assert t.on_gpu is False

    def test_from_ndarray(self):
        arr = np.array([4.0, 5.0], dtype=np.float64)
        t = Tensor(arr)
        assert t.dtype == np.float32
        np.testing.assert_array_equal(t.data, arr.astype(np.float32))

    def test_from_scalar_int(self):
        t = Tensor(7)
        assert t.is_constant is True
        assert t.const_value == 7.0
        assert t.shape == (1,)

    def test_from_scalar_float(self):
        t = Tensor(3.14)
        assert t.is_constant is True
        assert abs(t.const_value - 3.14) < 1e-5

    def test_lazy_node_shape_inferred(self):
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])
        c = Tensor(None, op="add", inputs=[a, b])
        assert c.shape == (2,)
        assert c.is_leaf is False
        assert c.op == "add"

    def test_repr_does_not_raise(self):
        t = Tensor([1, 2, 3])
        assert "Tensor" in repr(t)


# ---------------------------------------------------------------------------
# Operator overloading (lazy graph construction)
# ---------------------------------------------------------------------------

class TestOperatorOverloading:
    def test_add_returns_tensor(self):
        a = Tensor([1.0])
        b = Tensor([2.0])
        c = a + b
        assert isinstance(c, Tensor)
        assert c.op == "add"
        assert c.is_leaf is False

    def test_sub_returns_tensor(self):
        a = Tensor([5.0])
        b = Tensor([3.0])
        c = a - b
        assert c.op == "sub"

    def test_mul_returns_tensor(self):
        a = Tensor([2.0])
        b = Tensor([3.0])
        c = a * b
        assert c.op == "mul"

    def test_div_returns_tensor(self):
        a = Tensor([6.0])
        b = Tensor([2.0])
        c = a / b
        assert c.op == "div"

    def test_scalar_rhs_wrapped(self):
        a = Tensor([1.0, 2.0])
        c = a + 5
        assert c.op == "add"
        assert c.inputs[1].is_constant is True
        assert c.inputs[1].const_value == 5.0


# ---------------------------------------------------------------------------
# CPU operations
# ---------------------------------------------------------------------------

class TestCPUOps:
    def test_add_returns_tensor(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]))
        b = Tensor(np.array([4.0, 5.0, 6.0]))
        result = nx.add(a, b)
        assert isinstance(result, Tensor)
        np.testing.assert_array_almost_equal(result.data, [5.0, 7.0, 9.0])

    def test_sub_returns_tensor(self):
        a = Tensor(np.array([10.0, 20.0]))
        b = Tensor(np.array([3.0, 5.0]))
        result = nx.sub(a, b)
        assert isinstance(result, Tensor)
        np.testing.assert_array_almost_equal(result.data, [7.0, 15.0])

    def test_mul_returns_tensor(self):
        a = Tensor(np.array([2.0, 3.0]))
        b = Tensor(np.array([4.0, 5.0]))
        result = nx.mul(a, b)
        assert isinstance(result, Tensor)
        np.testing.assert_array_almost_equal(result.data, [8.0, 15.0])

    def test_div_returns_tensor(self):
        a = Tensor(np.array([10.0, 9.0]))
        b = Tensor(np.array([2.0, 3.0]))
        result = nx.div(a, b)
        assert isinstance(result, Tensor)
        np.testing.assert_array_almost_equal(result.data, [5.0, 3.0])

    def test_div_by_zero_raises(self):
        a = Tensor(np.array([1.0, 2.0]))
        b = Tensor(np.array([0.0, 1.0]))
        with pytest.raises(ZeroDivisionError):
            nx.div(a, b)

    def test_result_is_float32(self):
        a = Tensor(np.array([1.0, 2.0]))
        b = Tensor(np.array([3.0, 4.0]))
        result = nx.add(a, b)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# eval() — CPU path
# ---------------------------------------------------------------------------

class TestEvalCPU:
    def test_eval_add(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]))
        b = Tensor(np.array([4.0, 5.0, 6.0]))
        c = a + b
        result = c.eval()
        assert isinstance(result, Tensor)
        np.testing.assert_array_almost_equal(result.data, [5.0, 7.0, 9.0])

    def test_eval_sub(self):
        a = Tensor(np.array([10.0, 8.0]))
        b = Tensor(np.array([3.0, 2.0]))
        result = (a - b).eval()
        np.testing.assert_array_almost_equal(result.data, [7.0, 6.0])

    def test_eval_mul(self):
        a = Tensor(np.array([2.0, 3.0]))
        b = Tensor(np.array([5.0, 4.0]))
        result = (a * b).eval()
        np.testing.assert_array_almost_equal(result.data, [10.0, 12.0])

    def test_eval_div(self):
        a = Tensor(np.array([12.0, 9.0]))
        b = Tensor(np.array([4.0, 3.0]))
        result = (a / b).eval()
        np.testing.assert_array_almost_equal(result.data, [3.0, 3.0])

    def test_eval_chain(self):
        a = Tensor(np.array([1.0, 2.0]))
        b = Tensor(np.array([3.0, 4.0]))
        c = Tensor(np.array([5.0, 6.0]))
        result = ((a + b) * c).eval()
        np.testing.assert_array_almost_equal(result.data, [20.0, 36.0])

    def test_eval_leaf_returns_self(self):
        a = Tensor(np.array([1.0]))
        assert a.eval() is a

    def test_eval_returns_tensor(self):
        a = Tensor(np.array([1.0]))
        b = Tensor(np.array([2.0]))
        result = (a + b).eval()
        assert isinstance(result, Tensor)


# ---------------------------------------------------------------------------
# Constant folding
# ---------------------------------------------------------------------------

class TestConstantFolding:
    def test_fold_two_constants(self):
        a = Tensor(3.0)
        b = Tensor(4.0)
        c = Tensor(None, op="add", inputs=[a, b])
        folded = c._fold_constants()
        assert folded.is_constant is True
        assert abs(folded.const_value - 7.0) < 1e-5

    def test_fold_mul(self):
        a = Tensor(2.0)
        b = Tensor(5.0)
        c = Tensor(None, op="mul", inputs=[a, b])
        folded = c._fold_constants()
        assert abs(folded.const_value - 10.0) < 1e-5

    def test_non_constant_not_folded(self):
        a = Tensor(np.array([1.0]))
        b = Tensor(2.0)
        c = Tensor(None, op="add", inputs=[a, b])
        folded = c._fold_constants()
        assert folded.is_constant is False
        assert folded.op == "add"


# ---------------------------------------------------------------------------
# _build_fused — expression string generation
# ---------------------------------------------------------------------------

class TestBuildFused:
    def test_simple_add(self):
        a = Tensor(np.array([1.0, 2.0]))
        b = Tensor(np.array([3.0, 4.0]))
        node = Tensor(None, op="add", inputs=[a, b])
        expr, leaves = node._build_fused()
        assert "+" in expr
        assert len(leaves) == 2

    def test_constant_inlined(self):
        a = Tensor(np.array([1.0, 2.0]))
        c = Tensor(3.0)
        node = Tensor(None, op="mul", inputs=[a, c])
        expr, leaves = node._build_fused()
        assert "3.00000000f" in expr
        assert len(leaves) == 1  # constant not in leaves

    def test_chained_ops(self):
        a = Tensor(np.array([1.0]))
        b = Tensor(np.array([2.0]))
        c = Tensor(np.array([3.0]))
        inner = Tensor(None, op="add", inputs=[a, b])
        outer = Tensor(None, op="mul", inputs=[inner, c])
        expr, leaves = outer._build_fused()
        assert expr is not None
        assert len(leaves) == 3

    def test_multiply_add_uses_fmaf(self):
        a = Tensor(np.array([1.0]))
        b = Tensor(np.array([2.0]))
        c = Tensor(np.array([3.0]))
        mul = Tensor(None, op="mul", inputs=[a, b])
        node = Tensor(None, op="add", inputs=[mul, c])
        expr, leaves = node._build_fused()
        assert expr.startswith("fmaf(")
        assert len(leaves) == 3


# ---------------------------------------------------------------------------
# Memory pool
# ---------------------------------------------------------------------------

class TestMempool:
    def test_pool_size_starts_empty(self):
        mempool.clear_pool()
        assert mempool.pool_size() == 0

    def test_clear_pool(self):
        mempool.clear_pool()
        assert mempool.pool_size() == 0

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_alloc_and_free(self):
        mempool.clear_pool()
        ptr = mempool.alloc(1024)
        assert ptr is not None
        mempool.free(ptr, 1024)
        assert mempool.pool_size() == 1024

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_reuse_from_pool(self):
        mempool.clear_pool()
        ptr1 = mempool.alloc(512)
        mempool.free(ptr1, 512)
        ptr2 = mempool.alloc(512)
        assert ptr2 is ptr1  # reused

    def test_free_null_ptr_no_error(self):
        mempool.free(None, 0)  # should not raise


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

class TestDeviceSelection:
    def test_set_cpu(self, capsys):
        nx.set_default_device("cpu")
        from novax import dispatch
        assert dispatch.DEFAULT_DEVICE == "cpu"

    def test_set_gpu(self, capsys):
        nx.set_default_device("gpu")
        from novax import dispatch
        assert dispatch.DEFAULT_DEVICE == "gpu"
        nx.set_default_device("cpu")  # restore

    def test_cpu_ops_work_when_device_cpu(self):
        nx.set_default_device("cpu")
        a = Tensor(np.array([1.0, 2.0]))
        b = Tensor(np.array([3.0, 4.0]))
        result = nx.add(a, b)
        assert isinstance(result, Tensor)
        np.testing.assert_array_almost_equal(result.data, [4.0, 6.0])


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_enter_returns_tensor(self):
        t = Tensor(np.array([1.0]))
        with t as ctx:
            assert ctx is t

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_exit_frees_gpu_memory(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]))
        t.to_gpu()
        assert t.on_gpu is True
        with t:
            pass
        assert t.on_gpu is False
        assert t.gpu_ptr is None


# ---------------------------------------------------------------------------
# GPU round-trip (skipped when no GPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestGPUOps:
    def test_to_gpu_and_to_host(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = Tensor(arr)
        t.to_gpu()
        assert t.on_gpu is True
        assert t.data is None
        result = t.to_host()
        np.testing.assert_array_almost_equal(result, arr)

    def test_to_gpu_idempotent(self):
        t = Tensor(np.array([1.0, 2.0]))
        t.to_gpu()
        ptr_before = t.gpu_ptr
        t.to_gpu()
        assert t.gpu_ptr is ptr_before

    def test_gpu_add_via_dispatch(self):
        a = Tensor(np.array([1.0, 2.0, 3.0])).to_gpu()
        b = Tensor(np.array([4.0, 5.0, 6.0])).to_gpu()
        nx.set_default_device("gpu")
        result = nx.add(a, b)
        assert isinstance(result, Tensor)
        np.testing.assert_array_almost_equal(result.to_host(), [5.0, 7.0, 9.0])

    def test_eval_gpu_add(self):
        a = Tensor(np.array([1.0, 2.0])).to_gpu()
        b = Tensor(np.array([3.0, 4.0])).to_gpu()
        result = (a + b).eval()
        assert isinstance(result, Tensor)
        np.testing.assert_array_almost_equal(result.to_host(), [4.0, 6.0])

    def test_eval_gpu_sub(self):
        a = Tensor(np.array([5.0, 6.0])).to_gpu()
        b = Tensor(np.array([1.0, 2.0])).to_gpu()
        result = (a - b).eval()
        np.testing.assert_array_almost_equal(result.to_host(), [4.0, 4.0])

    def test_free_releases_to_pool(self):
        t = Tensor(np.array([1.0])).to_gpu()
        t.free()
        assert t.on_gpu is False
        assert t.gpu_ptr is None
