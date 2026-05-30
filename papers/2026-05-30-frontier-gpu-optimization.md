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
| [AutoKernel: Autonomous GPU Kernel Optimization via Iterative Agent-Driven Search](https://arxiv.org/abs/2603.21331) | A 2026 autonomous kernel loop profiles a model, ranks bottlenecks by Amdahl impact, then validates candidates through smoke tests, shape sweeps, numerical checks, determinism checks, and edge cases before recording speedups. | NovaX's loop should keep the current correctness-plus-benchmark gate, but add bottleneck ranking/profiling notes so experiments attack the cases that dominate focused geomean. |
| [TileLang: A Composable Tiled Programming Model for AI Systems](https://arxiv.org/abs/2504.17577) | Separating dataflow from scheduling gives a usable way to express tiled kernels while leaving thread binding, layout, tensorization, and pipelining tunable. | A future NovaX backend could generate TileLang/Triton-like kernels from focused graph patterns instead of hand-authoring each CUDA kernel. |
| [CUDA-LLM: LLMs Can Write Efficient CUDA Kernels](https://arxiv.org/abs/2506.09092) | Automated CUDA generation works better when correctness, compile success, and measured latency are jointly optimized through feedback. | Treat NovaX autoresearch as a feedback loop: generate one small kernel idea, run correctness, benchmark focused cases, log result, then revise. |
| [TritonForge: Profiling-Guided Framework for Automated Triton Kernel Optimization](https://arxiv.org/abs/2512.09196) | Profiling-guided iterative transformations are a practical path toward better Triton kernels. Runtime measurements drive the next edit. | Add optional profiling artifacts to future loops, especially for fusion and fused matmul cases, so changes target measured stalls rather than guesses. |
| [ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels](https://arxiv.org/abs/2511.13940) | Multi-GPU performance depends heavily on overlapping communication, scheduling, and data-transfer primitives. | Not an immediate single-GPU benchmark target, but relevant if NovaX expands beyond one GPU. |
| [CUDA-L2: Surpassing cuBLAS Performance for Matrix Multiplication through Reinforcement Learning](https://arxiv.org/abs/2512.02551) | The strongest matmul results come from automated search across a large kernel configuration space with measured execution speed as the reward. | NovaX should not expect static hand tweaks to beat cuBLAS/PyTorch broadly; serious matmul wins need a shape-keyed autotuning loop or vendor-library plan cache. |
| [tritonBLAS: Triton-based Analytical Approach for GEMM Kernel Parameter Selection](https://arxiv.org/abs/2512.04226) | Analytical models can predict near-optimal GEMM tiling from hardware/cache/shape parameters without paying full empirical autotuning cost. | For NovaX fused-mm, use shape-keyed analytical defaults before brute-force search; the 128 and 256 benchmark shapes likely need different tile/epilogue tactics. |
| [OptiML: An End-to-End Framework for Program Synthesis and CUDA Kernel Optimization](https://arxiv.org/abs/2602.12305) | Kernel optimization is framed as search under verification, with profiler-aware rewards guiding edits. | Fast-math substitutions should be treated as benchmarked candidates, not assumed wins; NovaX needs confirmation runs and correctness/latency gates for approximate math. |
| [FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection](https://arxiv.org/abs/2512.12949) | Large fusion wins come from reducing memory traffic with hardware-aware data movement and scheduling, not just syntactically combining operators. | NovaX's fused-mm edge needs a real tiled epilogue/fusion strategy; descriptor or stream-state micro-caching is unlikely to be enough. |
| [CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs](https://arxiv.org/abs/2605.19269) | A 2026 result argues that many memory-bound transformer side operations should execute as composable GEMM epilogues while GEMM output tiles are still on chip. | NovaX's fused-mm path should move toward true GEMM-epilogue fusion, not GEMM followed by a separate epilogue kernel. |
| [Fusing Epilog Operations with Matrix Multiplication Using nvmath-python](https://developer.nvidia.com/blog/fusing-epilog-operations-with-matrix-multiplication-using-nvmath-python/) | NVIDIA's current guidance for bias/ReLU matmul fusion is to use library epilogues that perform the post-op inside the matmul plan, avoiding extra traffic and launches. | NovaX should not expect manually skipping a zero bias load inside its naive CUDA tile to close the fused-mm gap; the route is true library or generated epilogues. |
| [Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks](https://arxiv.org/abs/2507.10789) | Blackwell exposes major performance through newer Tensor Core paths and memory hierarchy behavior; kernel choices must map to those hardware units. | For square GEMM, NovaX should prefer vendor Tensor Core math modes over hand-written FP32 CUDA tiles when accuracy policy allows it. |
| [Microbenchmark-Driven Analytical Performance Modeling Across Modern GPU Architectures](https://arxiv.org/abs/2605.04178) | Modern GPU performance work benefits from microbenchmark-grounded models of Tensor Cores and memory hierarchy rather than generic rules of thumb. | When NovaX sees a shape-stable GEMM case, first test hardware-backed library modes and shape gates before writing another kernel. |
| [Evaluating CUDA Tile for AI Workloads on Hopper and Blackwell GPUs](https://arxiv.org/abs/2604.23466) | Modern tile-centric APIs try to expose Tensor Core and TMA efficiency through a higher-level programming model, but performance still depends on mapping the tile schedule to the workload. | NovaX fusion work should move toward explicit tile/schedule choices; adding low-level qualifiers to the existing scalar-per-element kernel is too shallow. |
| [Nautilus: An Auto-Scheduling Tensor Compiler for Efficient Tiled GPU Kernels](https://arxiv.org/abs/2604.14825) | A 2026 tensor compiler result shows that expression rewrites, high-level transformations, and tile optimizations need to be searched jointly rather than hand-applied one at a time. | NovaX's next meaningful gains likely need a small IR plus scheduler/autotuner; isolated hand-specialized kernels have repeatedly failed the focused gate. |
| [GPUOS: A GPU Operating System Primitive for Transparent Operation Fusion](https://arxiv.org/abs/2604.17861) | A 2026 runtime direction for many small tensor ops is to avoid repeated host launches by using persistent GPU-side operation injection. | NovaX's Python overhead ceiling likely needs either larger graph capture regions or a lower-level runtime loop; individual Python-call optimizations are not enough. |
| [Hybrid JIT-CUDA Graph Optimization for Low-Latency Large Language Model Inference](https://arxiv.org/abs/2604.23467) | High-performance low-latency inference partitions static work into CUDA Graph replay and dynamic work into JIT-compiled kernels, reducing launch overhead and latency variance. | NovaX should keep pushing static graph replay, but pure Python replay loops are unlikely to be enough; the replay loop needs to move lower than Python or be amortized by larger captured regions. |
| [Boosting Performance of Iterative Applications on GPUs: Kernel Batching with CUDA Graphs](https://arxiv.org/abs/2501.09398) | Iterative launch-bound applications can batch multiple iterations by unrolling them into one CUDA Graph, reducing per-iteration launch overhead. | A `capture_many` style API is plausible for repeated fixed-shape NovaX workloads, but the benchmark gate needs stable evidence that only the intended graph path changes. |
| [FuseFlow: A Fusion-Centric Compilation Framework](https://weiya711.github.io/publications/asplos2026fuseflow.pdf) | Fusion schedules can be limited by over-fusion and under-fusion; profitable fusion depends on ordering and dataflow, not just combining adjacent operators. | NovaX's direct expression fusion is useful, but future wins need fusion planning around data movement and scheduling rather than more front-end string-building shortcuts. |

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

Experiment note:

- `1086ca8` added fixed kernels for the two current fusion-chain benchmarks.
  The focused gate failed and the intended fusion targets did not improve
  enough. Future primitive lowering should not just special-case expression
  strings; it needs scheduling/profiling evidence or a broader graph executor.
- `4a31903` retested direct fused-expression building under the focused metric,
  falling back to recursive constant folding only when direct lowering failed.
  It improved `fusion_chain5_n1000000` on both runs, but failed the focused
  gate twice due focused regressions elsewhere. Future fusion work should move
  below Python expression construction into graph lowering, scheduling, or
  kernel generation.

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

Experiment note:

- `86de151` tested a shape-gated cuBLAS GEMM followed by a separate bias+ReLU
  epilogue kernel for large fused-mm shapes. It improved the 256 fused-mm cases
  by about 1.19x to 1.23x, but failed the focused regression gate on rerun.
  Next attempt should avoid the separate epilogue launch, likely through
  cuBLASLt/CUTLASS-style fused epilogues or a better tile kernel.
- `1b04e93` tested a real cuBLASLt `RELU_BIAS` fused epilogue. Correctness
  passed and the 256 fused-mm cases improved, but the focused gate still failed
  with 7 regressions. The next vendor-kernel attempt should cache planning more
  aggressively or move to a persistent shape-specific wrapper outside the hot
  Python loop; otherwise the descriptor/heuristic surface eats the win.
- `c5e8268` tested a 16x32 rectangular tile for the direct fused
  matmul+bias+ReLU primitive. Correctness passed, but the focused gate saw zero
  improvements and eight regressions. The current 16x16 tile remains the better
  hand-written CUDA baseline for these shapes; future fused-mm work should use
  profile/autotune search rather than another static tile guess.
- `0c8c0ac` enabled TF32 tensor math for the square cuBLAS path at 256x256 and
  larger. It qualified twice. The primary run improved `matmul_512` by 1.59x,
  `matmul_1024` by 1.52x, and `matmul_256` by 1.23x while keeping the 128x128
  precision-sensitive test path on default math. This supports the hardware
  mapping lesson: for stable square GEMM, using Tensor Core-capable vendor
  modes beats another static CUDA tile.
- `9f22443` routed the larger fused matmul+bias+ReLU case through TF32 cuBLAS
  followed by an in-place epilogue kernel. It improved the intended fused-mm
  target and qualified once, but failed two of three runs. This supports CODA's
  lesson: the next serious fused-mm attempt needs a true GEMM epilogue while
  accumulator tiles are still hot, likely via cuBLASLt/CUTLASS-style epilogues
  with cached descriptors, not a separate launch.

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

Experiment note:

- `c830f9a` retested prepared PyCUDA calls as a launch-overhead reduction.
  Focused benchmark result: 0 improvements, 6 regressions. This reinforces that
  launch-wrapper micro-optimizations are not enough; future work needs profiling
  and larger graph/kernel changes.
- `77ed4e7` added `CUDAGraph.replay_many()` and updated the captured-inference
  benchmark to use it. The focused gate still failed and captured inference was
  slower than the saved best. This supports the hybrid JIT/CUDA Graph lesson:
  replay batching needs to move below Python or capture larger regions.
- `5626de1` added a `CUDAGraph.capture_many()` API and captured all 300 fixed
  repeated-inference passes inside one graph replay. Captured inference improved
  twice by about 1.14x and 1.19x versus the saved best, but the experiment still
  failed the focused gate on unrelated benchmark-case regressions. The idea
  remains aligned with CUDA graph kernel batching, but needs a more stable gate
  or isolated graph-path benchmark before it can be kept.
- `9a904b1` cached the cuBLAS stream binding so stable square-GEMM calls would
  skip repeated `cublasSetStream` ctypes calls. It improved captured inference
  only, while regressing five focused cases. This supports the CUDA-L2 and
  FlashFuser lesson: matmul/fused-mm progress needs searched kernel plans or
  real fusion dataflow changes, not small library state micro-caches.
- `a8d9886` used `__expf` only in non-leaf fused sigmoid expressions. It
  improved `fusion_chain5_n1000000` once, then failed confirmation with the
  chain itself regressing. This supports the OptiML lesson: approximate math
  edits must be treated as verified search candidates rather than obvious
  optimizations.
- `dfe4345` added `__restrict__` pointer qualifiers to generic fused
  elementwise kernels. It produced no focused improvements and regressed
  fused-mm cases. This supports the CUDA Tile/Nautilus lesson: NovaX needs
  explicit scheduling or tile-level changes for fusion, not shallow compiler
  hinting on the same scalar kernel shape.
- `29a37e2` tracked all-zero GPU tensors and skipped bias loads/adds in the
  fused matmul+ReLU kernel when the bias was zero. The focused benchmark still
  had zero improvements and five regressions. This supports the epilogue-fusion
  lesson: the cost is not the scalar bias load; the larger issue is using a
  naive FP32 tile instead of a Tensor Core GEMM with true epilogue fusion.
- `4a31903` retested direct fused-expression building as a Python front-end
  optimization. The result was directionally good for `fusion_chain5`, but not
  strong enough to pass the focused gate. This supports the AutoKernel/FuseFlow
  lesson: keep ranking bottlenecks and optimize lower-level schedules rather
  than spending many turns on Python-only expression-builder shortcuts.

## Things Not To Repeat Blindly

- Broad eager elementwise micro-optimizations without a focused-path win.
- General rectangular cuBLAS routing without shape gates.
- cuBLAS state micro-caching unless the matmul or fused-mm targets themselves
  improve.
- Static fused-mm tile changes without profiling or an autotune search.
- GEMM plus a separate epilogue launch as a substitute for true epilogue fusion.
- Fast-math substitutions without repeatable target-case wins.
- Pointer qualifier or signature-only fused-kernel tweaks.
- Zero-bias special cases inside the naive fused-mm tile.
- Softmax fast-math changes that do not improve the softmax benchmark itself.
- GPU backward rewrites that touch unrelated eager/autograd behavior.
- CUDA graph replay micro-tweaks unless captured inference itself improves.
- Direct fused-expression builder shortcuts unless the full focused gate stays
  green across confirmation runs.
