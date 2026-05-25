"""
Tier 2 — whole-subtree elementwise/activation fusion.

`eval()` previously fused only from a *binary* root and materialized every
intermediate, so a chain like `tanh(exp(a)+relu(b*c))` ran as 2-3 kernels plus
redundant child evaluations. `_fuse_template` now collapses the entire
elementwise/activation subtree into a single kernel expression; non-fusable
nodes (matmul, reductions) become opaque leaves.

The template builder is pure (no GPU), so its generated CUDA expressions are
verified directly on CPU. End-to-end fused execution is GPU-only (skipped here).
"""

import pytest
import numpy as np

import novax as nx
from novax.core import Tensor, GPU_AVAILABLE


def _leaf(arr):
    return Tensor(np.asarray(arr, dtype=np.float32))


# ---------------------------------------------------------------------------
# Pure template generation
# ---------------------------------------------------------------------------

class TestFuseTemplate:
    """`_fuse_template` produces the right CUDA expression + leaf ordering."""

    def test_relu_mul_add_chain(self):
        a, b, c = _leaf([1.0]), _leaf([2.0]), _leaf([3.0])
        mul = Tensor(None, op="mul", inputs=[a, b])
        add = Tensor(None, op="add", inputs=[mul, c])
        relu = Tensor(None, op="relu", inputs=[add])
        template, leaves = relu._fuse_template()
        assert template == "fmaxf(0.0f, ((__L0__ * __L1__) + __L2__))"
        assert leaves == [a, b, c]

    def test_six_op_chain_single_expression(self):
        a, b, c = _leaf([1.0]), _leaf([2.0]), _leaf([3.0])
        exp_a = Tensor(None, op="exp", inputs=[a])
        bc = Tensor(None, op="mul", inputs=[b, c])
        relu_bc = Tensor(None, op="relu", inputs=[bc])
        add = Tensor(None, op="add", inputs=[exp_a, relu_bc])
        tanh = Tensor(None, op="tanh", inputs=[add])
        template, leaves = tanh._fuse_template()
        assert template == "tanhf((expf(__L0__) + fmaxf(0.0f, (__L1__ * __L2__))))"
        assert leaves == [a, b, c]

    def test_matmul_is_opaque_leaf(self):
        X, W, bias = _leaf([[1.0]]), _leaf([[1.0]]), _leaf([1.0])
        mm = Tensor(None, op="matmul", inputs=[X, W])
        add = Tensor(None, op="add", inputs=[mm, bias])
        relu = Tensor(None, op="relu", inputs=[add])
        template, leaves = relu._fuse_template()
        # matmul does not expand — it is a leaf placeholder
        assert template == "fmaxf(0.0f, (__L0__ + __L1__))"
        assert leaves == [mm, bias]

    def test_constants_folded_inline(self):
        a = _leaf([5.0])
        mul = Tensor(None, op="mul", inputs=[a, Tensor(2.0)])
        template, leaves = mul._fuse_template()
        assert template == "(__L0__ * 2.00000000f)"
        assert leaves == [a]

    def test_repeated_leaf_deduped(self):
        a = _leaf([5.0])
        mul = Tensor(None, op="mul", inputs=[a, a])
        template, leaves = mul._fuse_template()
        assert template == "(__L0__ * __L0__)"
        assert leaves == [a]

    def test_pow_emits_powf(self):
        a, b = _leaf([2.0]), _leaf([3.0])
        p = Tensor(None, op="pow", inputs=[a, b])
        template, leaves = p._fuse_template()
        assert template == "powf(__L0__, __L1__)"
        assert leaves == [a, b]


# ---------------------------------------------------------------------------
# requires_grad detection (gates the forward-only fusion path)
# ---------------------------------------------------------------------------

class TestAnyRequiresGrad:
    def test_no_grad_subtree(self):
        a, b = _leaf([1.0]), _leaf([2.0])
        node = Tensor(None, op="add", inputs=[a, b])
        assert node._any_requires_grad() is False

    def test_grad_leaf_detected_through_graph(self):
        a = Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
        b = _leaf([2.0])
        mul = Tensor(None, op="mul", inputs=[a, b])
        node = Tensor(None, op="relu", inputs=[mul])
        assert node._any_requires_grad() is True


# ---------------------------------------------------------------------------
# CPU correctness: manual lazy graphs evaluate via the non-fused path
# ---------------------------------------------------------------------------

class TestLazyGraphCPUCorrectness:
    def test_relu_mul_add_matches_numpy(self):
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([0.5, 0.5, -1.0], dtype=np.float32)
        c = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        mul = Tensor(None, op="mul", inputs=[_leaf(a), _leaf(b)])
        add = Tensor(None, op="add", inputs=[mul, _leaf(c)])
        relu = Tensor(None, op="relu", inputs=[add])
        out = relu.eval()
        np.testing.assert_array_almost_equal(out.data, np.maximum(0.0, a * b + c), decimal=5)


# ---------------------------------------------------------------------------
# GPU end-to-end: the whole chain runs as one fused kernel and matches numpy
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestFullFuseGPU:
    def test_six_op_chain_values(self):
        np.random.seed(0)
        a = np.random.randn(4096).astype(np.float32)
        b = np.random.randn(4096).astype(np.float32)
        c = np.random.randn(4096).astype(np.float32)
        an, bn, cn = Tensor(a).to_gpu(), Tensor(b).to_gpu(), Tensor(c).to_gpu()
        out = nx.tanh(nx.exp(an) + nx.relu(nx.mul(bn, cn))).eval()
        ref = np.tanh(np.exp(a) + np.maximum(0.0, b * c))
        np.testing.assert_array_almost_equal(out.to_host(), ref, decimal=4)

    def test_fused_uses_single_launch(self):
        # A 5-op chain should compile to exactly one fused kernel.
        np.random.seed(1)
        a = np.random.randn(1024).astype(np.float32)
        b = np.random.randn(1024).astype(np.float32)
        c = np.random.randn(1024).astype(np.float32)
        an, bn, cn = Tensor(a).to_gpu(), Tensor(b).to_gpu(), Tensor(c).to_gpu()
        node = nx.sigmoid(nx.relu(nx.mul(an, bn) + cn) * an)
        fused = node._try_full_fuse()
        assert fused is not None
        ref = 1.0 / (1.0 + np.exp(-(np.maximum(0.0, a * b + c) * a)))
        np.testing.assert_array_almost_equal(fused.to_host(), ref, decimal=4)

    def test_grad_path_skips_fusion(self):
        # When a leaf requires grad, eval() must not take the fused forward path.
        a = Tensor(np.random.randn(256).astype(np.float32), requires_grad=True).to_gpu()
        node = nx.relu(a * a)
        assert node._any_requires_grad() is True
