# NovaX GPU Optimization Plan

> Goal: close (and then beat) the GPU performance gap against PyTorch.
> Status legend: ⬜ not started · 🟡 in progress · ✅ done

---

## Context

After the first round of GPU work (cuBLAS matmul, CUDA Graphs, fp16, Triton
softmax, pattern-fused matmul+bias+relu), a fresh GPU benchmark on a real
device showed:

| Section            | NovaX vs PyTorch          |
|--------------------|---------------------------|
| Matrix mult (large)| ≈ tied (~50 ms @ 4096³)   |
| Matrix mult (small)| 🔴 loses                  |
| Elementwise        | 🔴 ~3–5× slower           |
| Activations        | 🔴 ~6× slower             |
| Reductions         | 🔴 ~5× slower             |
| Kernel fusion      | 🔴 ~4–6× slower           |
| MLP fwd / inference| ❌ crashed (size mismatch)|

The shape of the result — *lose on small, tie on huge* — is the fingerprint of
**per-launch CPU overhead**, not slow kernels. cuBLAS already does the heavy
GEMM lifting; everything else is dominated by what happens *around* each kernel
launch.

### Honest ceiling on matmul

PyTorch's matmul **is** cuBLAS. We will not beat cuBLAS with cuBLAS — parity is
the ceiling for raw GEMM. The wins come on the axes where PyTorch eager mode
*also* pays a tax: **launch overhead, fusion, and whole-graph replay.**

---

## Root-cause analysis (grounded in the code)

1. **A driver query on every kernel launch.**
   `_optimal_block_size()` (`ops/launcher.py`) calls
   `cuda.Device(0).get_attribute(...)` *per launch*; `_get_stream()` calls
   `cuda.Context.get_current()` *per launch*. These are driver round-trips that
   dwarf a ~5 µs elementwise kernel → explains the flat 5–6× tax.

2. **cuBLAS handle created and destroyed every matmul.**
   `_launch_matmul_cublas()` calls `cublasCreate()` + `cublasDestroy()` on every
   call. Handle creation allocates workspace → small matmuls lose, 4096³ ties.

3. **Partial fusion, re-derived every eval.**
   `_build_fused` only fires when the graph *root* is a binary op
   (`core.py: eval()`); a unary root (`tanh(...)`) splits into multiple kernels.
   `_fold_constants()` + `_build_fused()` rebuild the whole graph in Python on
   every `eval()`.

4. **Forward GPU binary path can't broadcast (the crash).**
   `nx.matmul(X, W)` runs *eagerly* (concrete leaves), so `matmul + bias` becomes
   `add(concrete_mm, bias)` whose matmul child is `op=None` → the
   `_match_matmul_bias` auto-fusion pattern never fires. It falls to
   `launch_fused([mm(M·N), bias(N)], …)` whose `assert t.size == n` throws
   *"All inputs must have same size"*. This is the MLP-forward / inference crash.

---

## Tier 0 — Correctness: broadcasting in the forward GPU binary path

**Why first:** unblocks the MLP-forward and inference benchmarks (currently
crashing), and is a prerequisite for any honest end-to-end comparison.

- [ ] Add `launch_broadcast_binary(big, small, op_symbol, small_is_left, op_name)`
      to `ops/launcher.py`: kernel computes `out[idx] = a[idx] OP b[idx % m]`
      (row/trailing-dim broadcast), correct for `(M,N) + (N,)` bias adds in
      row-major layout.
- [ ] In `core.py::_eval_binary`, only enter the fused path when **all leaves
      share one size**; otherwise, when `left.size != right.size`, route to a new
      `_eval_broadcast_binary` (trailing-dim broadcast, else CPU fallback).
- [ ] Autograd already unbroadcasts (`autograd.py::_unbroadcast`) — no change
      needed; verify gradient shapes for `matmul + bias`.

**Files:** `ops/launcher.py`, `core.py`
**Verify:** new GPU-skipped tests for shape/decision + CPU correctness;
re-run notebook cells 8 (MLP fwd) and 13 (inference) on GPU.

---

## Tier 1 — Kill per-launch CPU overhead

**Why:** removes the flat 5–6× tax across elementwise / activations / reductions,
and makes small matmul competitive. Cheapest, highest-leverage change.

- [ ] **Cache device properties once.** `_optimal_block_size()` resolves the
      block size a single time into a module global, then returns it directly.
- [ ] **Cache the stream.** `_get_stream()` queries `Context.get_current()` only
      until the stream is created, then returns the cached stream (capture stream
      still takes priority via a cheap global check).
- [ ] **Persistent cuBLAS handle.** Create one handle lazily via
      `_get_cublas_handle()`, reuse across calls, set its stream per-call
      (`cublasSetStream`, cheap), destroy at exit. Removes create/destroy from the
      matmul hot path.

**Files:** `ops/launcher.py`
**Verify:** existing suite stays green; re-benchmark — expect elementwise /
activations / reductions to drop toward parity, small matmul to improve.

---

## Tier 2 — Real whole-graph fusion

**Why:** fixes the "kernel fusion" chart and activations; this is NovaX's
structural advantage over stock PyTorch eager.

- [x] **Fuse through unary roots.** `_try_full_fuse` (`core.py`) compiles the
      maximal elementwise/activation subtree into one kernel regardless of root
      op; non-fusable nodes (matmul, reductions, softmax) become evaluated leaves.
      Gated on no-grad so the per-op path still attaches backward closures.
- [x] **Broadcast-aware fusion.** `launch_fused` now sizes output to the largest
      leaf and the template indexes broadcast leaves as `x{i}[idx % size_i]`.
- [x] **Grid-stride loop** in the fused kernel (valid for any element count).
- [ ] **Memoize the compiled expression** (fused source + leaf order) on the node
      so repeated `eval()` in a loop skips Python re-derivation. *(remaining)*
- [ ] **Vectorized memory access** (`float4` / `half2`) in generated elementwise
      kernels → up to ~2–4× on memory-bound ops. *(remaining)*

**Files:** `core.py`, `ops/launcher.py`
**Done:** full-subtree fusion + broadcast-aware leaves + grid-stride, verified by
`tests/test_fusion_t2.py` (pure template-generation tests + GPU end-to-end).

---

## Tier 3 — Frontier GEMM + reductions

**Why:** where NovaX can pull ahead of a naive eager op sequence.

- [ ] **cuBLASLt fused epilogue** (GEMM + bias + ReLU) with **TF32 / Tensor
      Cores**, replacing the hand-written tiled `matmul_bias_relu`.
- [ ] **Warp-shuffle reductions** (`__shfl_down_sync`) + vectorized loads in a
      single grid-stride pass, replacing the 2–3 separate shared-memory kernels.
- [ ] **cuDNN (or keep Triton)** fast path for softmax / activations.

**Files:** `ops/launcher.py`, `ops/gpu/*`

---

## Tier 4 — Beat PyTorch eager: auto CUDA-Graph capture

**Why:** the real path to *winning*. PyTorch eager pays per-op Python + dispatch
cost every iteration; a captured NovaX graph replays a whole forward pass in
microseconds.

- [ ] Make `eval()` **auto-capture** a CUDA Graph keyed by expression structure
      and **replay** on repeated calls with identical shapes (reuse the existing
      `CUDAGraph`).
- [ ] Cache instantiated graphs; invalidate on shape change.

**Files:** `core.py`, `ops/launcher.py`

---

## Verification (each tier)

1. `pytest tests/ -q` — full suite green (CPU; GPU tests auto-skip here).
2. Re-run `benchmarks/novax_gpu_benchmark.ipynb` on a real GPU (Colab T4+).
3. Track the scoreboard (wins / ties / losses) tier over tier.

## Suggested order

T0 (correctness) → T1 (overhead) → T2 (fusion) → T4 (graph replay) → T3 (frontier GEMM).
T4 is sequenced before T3 because graph replay is the largest *relative* win for
the inference/training loop workloads NovaX targets.
