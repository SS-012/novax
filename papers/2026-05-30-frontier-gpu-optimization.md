# Frontier GPU Optimization Scan - 2026-05-30

This scan records current paper-driven ideas for NovaX optimization. The goal is
to turn literature into concrete autoresearch hypotheses, not to archive papers
for its own sake.

## Decision For NovaX

Focus optimization on NovaX's differentiated path:

- lazy elementwise graph fusion,
- CUDA graph capture/replay,
- square or otherwise stable matmul fast paths,
- fused matmul epilogues such as bias and activation,
- GPU-resident training paths once they can be isolated from broad regressions.

Do not spend the main loop trying to beat PyTorch on every isolated eager
elementwise or activation case. Keep those cases as guardrails.

## Papers Scanned

| Source | Main Learning | NovaX Hypothesis |
| --- | --- | --- |
| [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) | Modern wins come from IO-aware tiling plus overlapping computation and data movement. Hopper-specific techniques include warp specialization, interleaving block matmul and softmax, and FP8/block quantization. | Do not write naive "one more kernel" versions of compound ops. For future attention/MLP kernels, design around tiling, SRAM reuse, and overlap. |
| [ThunderKittens: Simple, Fast, and Adorable AI Kernels](https://arxiv.org/abs/2410.20399) | A small tile abstraction stack can match or beat expert kernels across AI operations by mapping cleanly to warp, block, and grid levels. | NovaX should consider a tiny internal tile-kernel layer for hot fused kernels instead of growing many unrelated ad hoc CUDA strings. |
| [Optimal Kernel Orchestration for Tensor Programs with Korch](https://arxiv.org/pdf/2406.09465) | Operator-level fusion is often too coarse. Decomposing operators into primitives and optimizing kernel orchestration can find strategies outside manual fusion rules. | NovaX's lazy graph should eventually lower to primitive graphs, not only expression strings. This is relevant to softmax, reductions, and matmul epilogues. |
| [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005) | High-performance serving combines JIT-specialized templates, memory-layout choices, load-balanced scheduling, and compatibility with CUDA Graph static requirements. | NovaX should make graph-captured static workloads first-class: fixed-shape graph keys, capture-safe memory reuse, and explicit replay APIs. |
| [KernelBench: Can LLMs Write Efficient GPU Kernels?](https://arxiv.org/abs/2502.10517) | LLM-generated kernels need correctness checks, execution feedback, and profiling feedback. Even frontier methods match PyTorch in fewer than 20 percent of cases without strong iteration. | Autoresearch should not trust plausible kernel edits. Every hypothesis needs tests, benchmark comparison, and preferably profiling evidence. |
| [TileLang: A Composable Tiled Programming Model for AI Systems](https://arxiv.org/abs/2504.17577) | Separating dataflow from scheduling gives a usable way to express tiled kernels while leaving thread binding, layout, tensorization, and pipelining tunable. | A future NovaX backend could generate TileLang/Triton-like kernels from focused graph patterns instead of hand-authoring each CUDA kernel. |
| [CUDA-LLM: LLMs Can Write Efficient CUDA Kernels](https://arxiv.org/abs/2506.09092) | Automated CUDA generation works better when correctness, compile success, and measured latency are jointly optimized through feedback. | Treat NovaX autoresearch as a feedback loop: generate one small kernel idea, run correctness, benchmark focused cases, log result, then revise. |
| [TritonForge: Profiling-Guided Framework for Automated Triton Kernel Optimization](https://arxiv.org/abs/2512.09196) | Profiling-guided iterative transformations are a practical path toward better Triton kernels. Runtime measurements drive the next edit. | Add optional profiling artifacts to future loops, especially for fusion and fused matmul cases, so changes target measured stalls rather than guesses. |
| [ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels](https://arxiv.org/abs/2511.13940) | Multi-GPU performance depends heavily on overlapping communication, scheduling, and data-transfer primitives. | Not an immediate single-GPU benchmark target, but relevant if NovaX expands beyond one GPU. |

## Cross-Paper Themes

The frontier is not "rewrite everything in C++" as a first principle. The
frontier pattern is:

1. Keep a flexible frontend.
2. Capture or lower hot regions into an IR.
3. Fuse across operation boundaries.
4. Generate tiled, hardware-aware kernels for stable shapes.
5. Use vendor primitives where they already dominate.
6. Autotune or profile-guided search instead of relying on static guesses.
7. Use CUDA Graph replay when the workload is static enough.

## NovaX Hypothesis Backlog

### H1: Focused CUDA graph path

Make `nx.CUDAGraph` easier to use for fixed-shape inference and training
fragments. Cache capture plans by shape and ensure memory pool behavior is
capture-safe.

Benchmark target:

- `inference_capture_300_passes`

Expected mechanism:

- remove Python, PyCUDA, and CUDA driver setup from repeated static execution.

### H2: Primitive graph lowering for fusion

Add an internal representation below lazy tensors where compound ops can be
split into primitives with compatible parallelism before kernel selection.

Benchmark targets:

- `fusion_chain3_n1000000`
- `fusion_chain5_n1000000`
- future softmax/reduction chains

Expected mechanism:

- avoid both over-fusion and under-fusion by selecting kernel boundaries
  intentionally.

### H3: Tile-based fused matmul epilogues

Replace generic fused matmul CUDA strings with a shape-gated tiled kernel layer
or a cuBLASLt/CUTLASS-backed path for bias/activation epilogues.

Benchmark targets:

- `fused_mm_naive_128_256_128`
- `fused_mm_linear_128_256_128`
- `fused_mm_naive_256_512_256`
- `fused_mm_linear_256_512_256`

Expected mechanism:

- reduce global-memory traffic and avoid separate epilogue launches while
  keeping matmul performance close to vendor kernels.

### H4: GPU-resident MLP backward, gated narrowly

Previous experiments made MLP backward beat PyTorch in one run, but broad
regressions killed the change. Revisit only with shape/graph gating so unrelated
eager paths are untouched.

Benchmark target:

- `mlp_forward_backward_256_128_64`

Expected mechanism:

- avoid host downloads for backward closures and keep gradients on device.

### H5: Profiling-guided kernel loop

For each future focused candidate, store the benchmark result plus a short
profiling note: launch count, transfer count, and likely bottleneck. Use that
note to choose the next experiment.

Benchmark targets:

- all focused differentiated cases

Expected mechanism:

- reduce noisy blind edits and converge toward measured bottlenecks.

## Things Not To Repeat Blindly

- Broad eager elementwise micro-optimizations without a focused-path win.
- General rectangular cuBLAS routing without shape gates.
- Softmax fast-math changes that do not improve the softmax benchmark itself.
- GPU backward rewrites that touch unrelated eager/autograd behavior.
- CUDA graph replay micro-tweaks unless captured inference itself improves.

