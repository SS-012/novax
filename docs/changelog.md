# Changelog

---

## [0.2.0] — 2026-05-24

### Added

**Math operations**

- `exp`, `log`, `sqrt`, `abs`, `neg` — unary elementwise ops with CPU (NumPy) and GPU (CUDA) backends
- `pow` — elementwise power (`a ** b`), binary, with full gradient support
- `__neg__`, `__pow__`, `__matmul__` operator overloads on `Tensor`
- Reverse operators: `__radd__`, `__rsub__`, `__rmul__`, `__rtruediv__`

**Activation functions**

- `relu` — `max(0, x)`, CUDA: `fmaxf(0.0f, a[idx])`
- `sigmoid` — `1 / (1 + exp(-x))`, CUDA: `1.0f / (1.0f + expf(-a[idx]))`
- `tanh` — CUDA: `tanhf(a[idx])`
- `softmax` — numerically stable, three-pass GPU implementation

**Reduction operations**

- `sum`, `mean`, `max`, `min` — reduce over all elements to a scalar `(1,)` tensor
- GPU: two-pass parallel tree reduction using 256-thread blocks with shared memory

**Linear algebra**

- `matmul` / `@` — matrix multiplication, CPU via `np.matmul`, GPU via 16×16 tiled CUDA kernel
- `launch_matmul_bias_relu` — fused matmul + bias + ReLU CUDA kernel (no extra memory round-trips)

**Automatic differentiation**

- `requires_grad` parameter on `Tensor`
- `Tensor.grad` — accumulated gradient after `backward()`
- `Tensor.backward()` — reverse-mode autodiff via topological sort
- `no_grad` context manager — disables gradient tracking in its scope
- Broadcasting gradient correction — `_unbroadcast()` sums gradients back to input shapes
- Gradient rules for all 19 ops

**Shape manipulation**

- `Tensor.reshape(*new_shape)` — reshape tensor, downloads from GPU if needed
- `Tensor.transpose(axes=None)` — NumPy-compatible axis transpose

**Infrastructure**

- Bucketed GPU memory pool — O(1) alloc/free using power-of-2 size bins (replaces linear scan)
- Adaptive CUDA block size — queries `MAX_THREADS_PER_BLOCK` per device, selects 512/256/128
- Kernel fusion extended to include unary ops (`exp`, `relu`, `sigmoid`, etc.) inside fused graphs
- `_is_lazy()` check in dispatch layer — unary and binary functions short-circuit correctly for lazy inputs

### Changed

- `eval()` refactored into `_eval_unary()`, `_eval_binary()`, `_eval_matmul()` helpers
- `to_gpu()` now guards against double-upload; clears host data from pool buffers
- `dispatch.py` module restructured: each function handles lazy path and eager path with grad setup

### Tests

- 101 tests passing; 13 GPU tests skipped when CUDA unavailable
- New `tests/test_phase1_ops.py` — coverage for all new ops, reshape, transpose, eval paths
- New `tests/test_autograd.py` — numerical gradient checks (finite differences) for all differentiable ops

---

## [0.1.16] — 2025

### Fixed

- CPU op dispatch fallback was missing for several ops; added explicit fallback chain
- Kernel cache was not keyed on CUDA source; identical names but different expressions could collide
- `to_gpu()` guard order was inverted — could attempt GPU upload without checking `GPU_AVAILABLE` first
- `eval()` binary GPU path was missing a `return` statement, falling through to CPU unnecessarily

### Added

- CI via GitHub Actions — Python 3.10/3.11/3.12 matrix with `pytest --cov`
- `tests/test_novax.py` — 46 tests covering creation, ops, eval, fusion, mempool, and GPU round-trips

---

## [0.1.0] — 2025

Initial release.

- `Tensor` class with CPU/GPU storage, lazy evaluation, and expression graph
- Binary elementwise ops: `add`, `sub`, `mul`, `div`
- GPU kernel launcher with PyCUDA, CUDA source-module cache
- Expression fusion — `_build_fused()` compiles multi-op chains into a single kernel
- GPU memory pool (`utils/mempool.py`)
- `set_default_device()` for runtime device selection
