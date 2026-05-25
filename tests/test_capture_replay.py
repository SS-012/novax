"""
Tests for the whole-graph capture/replay path and the per-eval overhead
reductions (cached Tensor-class resolver, cached grad-enabled accessor,
capture-safe memory pool recording).

CPU-runnable tests cover the decision/fallback logic; GPU tests (auto-skipped
without CUDA) verify capture + replay correctness end to end.
"""

import pytest
import numpy as np

import novax as nx
from novax.core import Tensor, GPU_AVAILABLE
from novax.ops import launcher
from novax.utils import mempool


# ---------------------------------------------------------------------------
# Floor-reduction helpers (Part 1)
# ---------------------------------------------------------------------------

class TestCachedResolvers:
    def test_tensor_cls_returns_tensor(self):
        assert launcher._tensor_cls() is Tensor

    def test_tensor_cls_is_cached(self):
        first = launcher._tensor_cls()
        second = launcher._tensor_cls()
        assert first is second
        assert launcher._Tensor is Tensor

    def test_grad_enabled_accessor(self):
        from novax.core import _grad_enabled
        # Default: gradient tracking is enabled.
        assert _grad_enabled() is True
        with nx.no_grad():
            assert _grad_enabled() is False
        assert _grad_enabled() is True


# ---------------------------------------------------------------------------
# Capture-safe memory pool recording (Part 2 plumbing)
# ---------------------------------------------------------------------------

class TestMempoolRecording:
    def test_record_state_machine(self):
        mempool.begin_record()
        assert mempool._recording is True
        out = mempool.end_record()
        assert mempool._recording is False
        assert isinstance(out, list)

    def test_end_record_returns_recorded_buckets(self):
        mempool.begin_record()
        # Simulate alloc bookkeeping without a device by appending buckets
        # exactly as alloc() would.
        mempool._record_buckets.append(mempool._bucket(4096))
        mempool._record_buckets.append(mempool._bucket(4096))
        buckets = mempool.end_record()
        assert buckets == [mempool._bucket(4096)] * 2

    def test_record_resets_between_sessions(self):
        mempool.begin_record()
        mempool._record_buckets.append(64)
        mempool.end_record()
        mempool.begin_record()
        assert mempool.end_record() == []


# ---------------------------------------------------------------------------
# Buffer recycling: __del__ returns GPU buffers to the caching pool
# ---------------------------------------------------------------------------

class TestBufferRecycling:
    def test_del_recycles_gpu_buffer(self, monkeypatch):
        calls = []
        monkeypatch.setattr(mempool, "free",
                            lambda ptr, nbytes, release=False: calls.append((ptr, nbytes)))
        t = Tensor(np.zeros(8, dtype=np.float32))
        # Pose as a GPU tensor holding a device buffer.
        sentinel = object()
        t.on_gpu = True
        t.gpu_ptr = sentinel
        t.size = 256
        t.dtype = np.float32
        t.__del__()
        assert len(calls) == 1
        assert calls[0][0] is sentinel
        assert calls[0][1] == 256 * 4   # 1024 bytes
        assert t.gpu_ptr is None         # buffer released from the tensor

    def test_del_is_noop_for_cpu_tensor(self, monkeypatch):
        calls = []
        monkeypatch.setattr(mempool, "free",
                            lambda *a, **k: calls.append(a))
        t = Tensor(np.zeros(8, dtype=np.float32))   # CPU tensor
        t.__del__()
        assert calls == []

    def test_del_skips_pinned_buffer(self, monkeypatch):
        calls = []
        monkeypatch.setattr(mempool, "free",
                            lambda *a, **k: calls.append(a))
        t = Tensor(np.zeros(8, dtype=np.float32))
        t.on_gpu = True
        t.gpu_ptr = object()
        t.size = 16
        t.pinned = True
        t.__del__()
        assert calls == []

    def test_del_fp16_byte_size(self, monkeypatch):
        calls = []
        monkeypatch.setattr(mempool, "free",
                            lambda ptr, nbytes, release=False: calls.append(nbytes))
        t = Tensor(np.zeros(8, dtype=np.float32))
        t.on_gpu = True
        t.gpu_ptr = object()
        t.size = 100
        t.dtype = np.float16
        t.__del__()
        assert calls == [200]   # 100 elements * 2 bytes


# ---------------------------------------------------------------------------
# CUDAGraph fallback behaviour (no GPU)
# ---------------------------------------------------------------------------

class TestCUDAGraphFallback:
    @pytest.mark.skipif(launcher.cuda is not None, reason="CPU-only assertion")
    def test_capture_requires_cuda(self):
        g = nx.CUDAGraph()
        with pytest.raises(RuntimeError):
            g.capture(lambda: None)

    def test_replay_without_capture_raises(self):
        g = nx.CUDAGraph()
        with pytest.raises(RuntimeError):
            g.replay()


# ---------------------------------------------------------------------------
# GPU: whole-graph capture + replay correctness
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestCaptureReplayGPU:
    def _mlp(self):
        np.random.seed(7)
        BS, IN, HID, OUT = 64, 32, 48, 16
        X  = Tensor(np.random.randn(BS, IN).astype(np.float32)).to_gpu()
        W1 = Tensor((np.random.randn(IN, HID) * 0.1).astype(np.float32)).to_gpu()
        b1 = Tensor(np.random.randn(HID).astype(np.float32)).to_gpu()
        W2 = Tensor((np.random.randn(HID, OUT) * 0.1).astype(np.float32)).to_gpu()
        b2 = Tensor(np.random.randn(OUT).astype(np.float32)).to_gpu()
        Xn, W1n, b1n = X.to_host(), W1.to_host(), b1.to_host()
        W2n, b2n = W2.to_host(), b2.to_host()
        ref = np.maximum(0.0, Xn @ W1n + b1n) @ W2n + b2n

        def fwd():
            h = nx.relu(nx.matmul(X, W1) + b1)
            return (nx.matmul(h, W2) + b2).eval()

        return fwd, ref

    def test_capture_returns_correct_output(self):
        fwd, ref = self._mlp()
        g = nx.CUDAGraph()
        out = g.capture(fwd)
        np.testing.assert_array_almost_equal(out.to_host(), ref, decimal=3)

    def test_replay_matches_reference(self):
        fwd, ref = self._mlp()
        g = nx.CUDAGraph()
        out = g.capture(fwd)
        for _ in range(5):
            g.replay()
            np.testing.assert_array_almost_equal(out.to_host(), ref, decimal=3)

    def test_replay_is_stable_across_calls(self):
        fwd, _ = self._mlp()
        g = nx.CUDAGraph()
        out = g.capture(fwd)
        g.replay(); r1 = out.to_host().copy()
        g.replay(); r2 = out.to_host().copy()
        np.testing.assert_array_equal(r1, r2)

    def test_elementwise_chain_capture(self):
        np.random.seed(8)
        a_np = np.random.randn(4096).astype(np.float32)
        b_np = np.random.randn(4096).astype(np.float32)
        c_np = np.random.randn(4096).astype(np.float32)
        a = Tensor(a_np).to_gpu(); b = Tensor(b_np).to_gpu(); c = Tensor(c_np).to_gpu()
        ref = np.tanh(np.exp(a_np) + np.maximum(0.0, b_np * c_np))

        def chain():
            return nx.tanh(nx.exp(a) + nx.relu(nx.mul(b, c))).eval()

        g = nx.CUDAGraph()
        out = g.capture(chain)
        np.testing.assert_array_almost_equal(out.to_host(), ref, decimal=4)
        for _ in range(3):
            g.replay()
            np.testing.assert_array_almost_equal(out.to_host(), ref, decimal=4)
