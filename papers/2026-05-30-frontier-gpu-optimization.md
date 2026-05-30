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
| [PyGraph: Robust Compiler Support for CUDA Graphs in PyTorch](https://arxiv.org/abs/2503.19779) | CUDA Graph gains become robust when capture is integrated into the compiler/runtime rather than exposed only as manual user code. | NovaX should treat graph capture as a runtime lowering path with shape keys, stable allocation, and compiler-like guards, not just a convenience wrapper. |
| [KernelBench: Can LLMs Write Efficient GPU Kernels?](https://arxiv.org/abs/2502.10517) | LLM-generated kernels need correctness checks, execution feedback, and profiling feedback. Even frontier methods match PyTorch in fewer than 20 percent of cases without strong iteration. | Autoresearch should not trust plausible kernel edits. Every hypothesis needs tests, benchmark comparison, and preferably profiling evidence. |
| [AutoKernel: Autonomous GPU Kernel Optimization via Iterative Agent-Driven Search](https://arxiv.org/abs/2603.21331) | A 2026 autonomous kernel loop profiles a model, ranks bottlenecks by Amdahl impact, then validates candidates through smoke tests, shape sweeps, numerical checks, determinism checks, and edge cases before recording speedups. | NovaX's loop should keep the current correctness-plus-benchmark gate, but add bottleneck ranking/profiling notes so experiments attack the cases that dominate focused geomean. |
| [CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization](https://arxiv.org/abs/2511.01884) | The agent loop explicitly combines candidate generation, correctness checks, hardware feedback such as Nsight Compute metrics, and repeated improvement. | NovaX should start attaching profiler counters to repeated failures around fused-mm and exact small GEMM; static edits are not enough once the target is near the noise floor. |
| [CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation](https://arxiv.org/abs/2602.24286) | The frontier is moving toward learned CUDA optimization skill with reliable verification and profiling rewards, not one-off prompt guesses. | Keep the autoresearch loop strict: correctness, focused latency, and regression guardrails should define reward. Longer-term, collect failed variants as training/search data. |
| [KernelBlaster: Continual Cross-Task CUDA Optimization via Memory-Augmented In-Context Reinforcement Learning](https://arxiv.org/abs/2602.14293) | A 2026 agentic CUDA optimizer stores prior optimization attempts in a persistent knowledge base and uses them to guide future edits. | NovaX should treat `autoresearch/results.tsv`, benchmark artifacts, and this paper log as memory: each failure should constrain the next search rather than being rediscovered. |
| [Model2Kernel: Model-Aware Symbolic Execution For Safe CUDA Kernels](https://arxiv.org/abs/2603.24595) | Recent kernel work emphasizes that model-specific shapes make CUDA memory safety subtle enough to require automated checking. | As NovaX moves toward generated or shape-specialized kernels, correctness tests need to expand with shape and bounds coverage; exact-shape kernels are fast but brittle. |
| [FlipFlop: A Static Analysis-based Energy Optimization Framework for GPU Kernels](https://arxiv.org/abs/2601.13345) | Static PTX analysis can narrow block-size choices and find throughput/power tradeoffs before exhaustive runtime search. | Future NovaX block-size experiments should be explicit autotune/profile sweeps, not isolated guesses like the failed 256-thread and coarsened fused kernels. |
| [Astra: A Multi-Agent System for GPU Kernel Performance Optimization](https://arxiv.org/abs/2509.07506) | Starting from existing CUDA kernels and iteratively applying loop transformations, memory-access changes, intrinsics, and fast math is a viable optimization workflow. | NovaX experiments should mutate the existing hot CUDA strings in small verifiable steps, but only keep transformations that survive full focused-suite validation. |
| [KernelAgent: Hardware-Guided GPU Kernel Optimization via Multi-Agent Orchestration](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/) | PyTorch's 2026 hardware-guided loop emphasizes profiler-derived bottleneck diagnosis before proposing kernel changes. | NovaX should increasingly attach profiling notes to experiments; pure Python launch-source caching can look plausible but still sit below benchmark noise. |
| [TileLang: A Composable Tiled Programming Model for AI Systems](https://arxiv.org/abs/2504.17577) | Separating dataflow from scheduling gives a usable way to express tiled kernels while leaving thread binding, layout, tensorization, and pipelining tunable. | A future NovaX backend could generate TileLang/Triton-like kernels from focused graph patterns instead of hand-authoring each CUDA kernel. |
| [CUDA-LLM: LLMs Can Write Efficient CUDA Kernels](https://arxiv.org/abs/2506.09092) | Automated CUDA generation works better when correctness, compile success, and measured latency are jointly optimized through feedback. | Treat NovaX autoresearch as a feedback loop: generate one small kernel idea, run correctness, benchmark focused cases, log result, then revise. |
| [TritonForge: Profiling-Guided Framework for Automated Triton Kernel Optimization](https://arxiv.org/abs/2512.09196) | Profiling-guided iterative transformations are a practical path toward better Triton kernels. Runtime measurements drive the next edit. | Add optional profiling artifacts to future loops, especially for fusion and fused matmul cases, so changes target measured stalls rather than guesses. |
| [ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels](https://arxiv.org/abs/2511.13940) | Multi-GPU performance depends heavily on overlapping communication, scheduling, and data-transfer primitives. | Not an immediate single-GPU benchmark target, but relevant if NovaX expands beyond one GPU. |
| [CUDA-L2: Surpassing cuBLAS Performance for Matrix Multiplication through Reinforcement Learning](https://arxiv.org/abs/2512.02551) | The strongest matmul results come from automated search across a large kernel configuration space with measured execution speed as the reward. | NovaX should not expect static hand tweaks to beat cuBLAS/PyTorch broadly; serious matmul wins need a shape-keyed autotuning loop or vendor-library plan cache. |
| [tritonBLAS: Triton-based Analytical Approach for GEMM Kernel Parameter Selection](https://arxiv.org/abs/2512.04226) | Analytical models can predict near-optimal GEMM tiling from hardware/cache/shape parameters without paying full empirical autotuning cost. | For NovaX fused-mm, use shape-keyed analytical defaults before brute-force search; the 128 and 256 benchmark shapes likely need different tile/epilogue tactics. |
| [FalconGEMM: Surpassing Hardware Peaks with Lower-Complexity Matrix Multiplication](https://arxiv.org/abs/2605.06057) | A 2026 GEMM framework combines code generation, group-parallel optimizations, and an analytical decision model to choose strategies by shape and hardware. | NovaX small-GEMM and fused-mm work should move toward generated candidate families with shape selection; one-off direct/shared-memory edits are too brittle. |
| [A Few Fit Most: Improving Performance Portability of SGEMM on GPUs using Multi-Versioning](https://arxiv.org/abs/2507.15277) | A single GEMM kernel rarely stays near-optimal across devices and shapes; a small set of generated variants can be more portable than one universal kernel. | NovaX fused-mm should move toward a shape-keyed kernel family, but variants need measured selection. A hand-picked exact-tile version without profiling is still just a static guess. |
| [OptiML: An End-to-End Framework for Program Synthesis and CUDA Kernel Optimization](https://arxiv.org/abs/2602.12305) | Kernel optimization is framed as search under verification, with profiler-aware rewards guiding edits. | Fast-math substitutions should be treated as benchmarked candidates, not assumed wins; NovaX needs confirmation runs and correctness/latency gates for approximate math. |
| [FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection](https://arxiv.org/abs/2512.12949) | Large fusion wins come from reducing memory traffic with hardware-aware data movement and scheduling, not just syntactically combining operators. | NovaX's fused-mm edge needs a real tiled epilogue/fusion strategy; descriptor or stream-state micro-caching is unlikely to be enough. |
| [CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs](https://arxiv.org/abs/2605.19269) | A 2026 result argues that many memory-bound transformer side operations should execute as composable GEMM epilogues while GEMM output tiles are still on chip. | NovaX's fused-mm path should move toward true GEMM-epilogue fusion, not GEMM followed by a separate epilogue kernel. |
| [Fusing Epilog Operations with Matrix Multiplication Using nvmath-python](https://developer.nvidia.com/blog/fusing-epilog-operations-with-matrix-multiplication-using-nvmath-python/) | NVIDIA's current guidance for bias/ReLU matmul fusion is to use library epilogues that perform the post-op inside the matmul plan, avoiding extra traffic and launches. | NovaX should not expect manually skipping a zero bias load inside its naive CUDA tile to close the fused-mm gap; the route is true library or generated epilogues. |
| [Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks](https://arxiv.org/abs/2507.10789) | Blackwell exposes major performance through newer Tensor Core paths and memory hierarchy behavior; kernel choices must map to those hardware units. | For square GEMM, NovaX should prefer vendor Tensor Core math modes over hand-written FP32 CUDA tiles when accuracy policy allows it. |
| [Microbenchmark-Driven Analytical Performance Modeling Across Modern GPU Architectures](https://arxiv.org/abs/2605.04178) | Modern GPU performance work benefits from microbenchmark-grounded models of Tensor Cores and memory hierarchy rather than generic rules of thumb. | When NovaX sees a shape-stable GEMM case, first test hardware-backed library modes and shape gates before writing another kernel. |
| [Evaluating CUDA Tile for AI Workloads on Hopper and Blackwell GPUs](https://arxiv.org/abs/2604.23466) | Modern tile-centric APIs try to expose Tensor Core and TMA efficiency through a higher-level programming model, but performance still depends on mapping the tile schedule to the workload. | NovaX fusion work should move toward explicit tile/schedule choices; adding low-level qualifiers to the existing scalar-per-element kernel is too shallow. |
| [Nautilus: An Auto-Scheduling Tensor Compiler for Efficient Tiled GPU Kernels](https://arxiv.org/abs/2604.14825) | A 2026 tensor compiler result shows that expression rewrites, high-level transformations, and tile optimizations need to be searched jointly rather than hand-applied one at a time. | NovaX's next meaningful gains likely need a small IR plus scheduler/autotuner; isolated hand-specialized kernels have repeatedly failed the focused gate. |
| [GPUOS: A GPU Operating System Primitive for Transparent Operation Fusion](https://arxiv.org/abs/2604.17861) | A 2026 runtime direction for many small tensor ops is to avoid repeated host launches by using persistent GPU-side operation injection. | NovaX's Python overhead ceiling likely needs either larger graph capture regions or a lower-level runtime loop; individual Python-call optimizations are not enough. |
| [Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs](https://arxiv.org/abs/2512.22219) | Persistent mega-kernels can lower tensor programs into SM-level task graphs, enabling cross-operator pipelining and decentralized scheduling inside one long-lived kernel. | NovaX's fusion path likely needs graph/runtime scheduling for chains and training blocks; per-kernel source tweaks are too local once the obvious launch wins are gone. |
| [ClusterFusion++: Expanding Cluster-Level Fusion to Full Transformer-Block Decoding](https://arxiv.org/abs/2604.23553) | Full decoder-block CUDA fusion reduces fragmented operator execution and repeated off-chip materialization across normalization, projections, attention, MLP, and residual paths. | NovaX should treat "beat PyTorch" as a block-level fusion problem. The differentiated edge is whole static regions, not isolated eager ops. |
| [Hybrid JIT-CUDA Graph Optimization for Low-Latency Large Language Model Inference](https://arxiv.org/abs/2604.23467) | High-performance low-latency inference partitions static work into CUDA Graph replay and dynamic work into JIT-compiled kernels, reducing launch overhead and latency variance. | NovaX should keep pushing static graph replay, but pure Python replay loops are unlikely to be enough; the replay loop needs to move lower than Python or be amortized by larger captured regions. |
| [Foundry: Template-Based CUDA Graph Context Materialization for Fast LLM Serving Cold Start](https://arxiv.org/abs/2604.06664) | CUDA Graphs are coupled to execution context, including device addresses and lazily loaded kernels; Foundry reduces online reconstruction cost by persisting graph topology plus context templates. | NovaX graph capture should treat memory layout and kernel warmup as first-class state. Replaying static regions is valuable, but capture-safe deterministic allocation matters as much as the replay API. |
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

Experiment note:

- `bea1152` routed the two exact repeated-inference rectangular GEMM shapes
  through TF32 cuBLAS. Correctness passed and `inference_capture_300_passes`
  improved once by about 1.17x, but seven focused rows regressed and the gate
  failed. The lesson is not that inference GEMM library routing is hopeless;
  it is that library-state or math-mode changes must be isolated and validated
  against square matmul/fused-mm guardrails, or moved behind a lower-level
  graph/runtime plan.

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
- `53d31b8` tested an exact 16x16 tiled variant for divisible shapes, removing
  boundary checks and unrolling the inner loop. Correctness passed, but the
  focused benchmark failed and the fused-mm targets did not improve enough to
  count. This reinforces the multi-versioning lesson: shape variants need
  measured selection, not just plausible simplification.
- `0c8c0ac` enabled TF32 tensor math for the square cuBLAS path at 256x256 and
  larger. It qualified twice. The primary run improved `matmul_512` by 1.59x,
  `matmul_1024` by 1.52x, and `matmul_256` by 1.23x while keeping the 128x128
  precision-sensitive test path on default math. This supports the hardware
  mapping lesson: for stable square GEMM, using Tensor Core-capable vendor
  modes beats another static CUDA tile.
- `d610a5c` added a 64x64-only tiled matmul kernel with no boundary checks and
  no runtime shape arguments. It qualified twice against the saved focused
  baseline, and a later tiebreaker still showed a 1.14x `matmul_64` win against
  the candidate-best run even though the stricter full-suite score failed on
  noise. This supports the multi-versioning lesson for very small fixed GEMM:
  narrow, exact-shape variants can be useful when they do not touch broader
  MLP or fused-mm paths.
- `9f22443` routed the larger fused matmul+bias+ReLU case through TF32 cuBLAS
  followed by an in-place epilogue kernel. It improved the intended fused-mm
  target and qualified once, but failed two of three runs. This supports CODA's
  lesson: the next serious fused-mm attempt needs a true GEMM epilogue while
  accumulator tiles are still hot, likely via cuBLASLt/CUTLASS-style epilogues
  with cached descriptors, not a separate launch.
- `ff3a8ac` narrowed cuBLASLt to an exact zero-bias `RELU` epilogue for
  `256x512x256`, avoiding the broader `RELU_BIAS` shape surface from
  `1b04e93`. The path was technically valid and qualified once, but failed
  confirmation with five focused regressions, including fusion-chain
  regressions. CODA remains the right direction, but a Python/ctypes cuBLASLt
  descriptor path is still too noisy; future epilogue work should use a
  persistent lower-level wrapper, a generated CUTLASS-style kernel, or an
  explicit plan cache outside the benchmark hot loop.
- `7c5ff68` tested a hand-written exact TF32 WMMA fused matmul+ReLU kernel for
  the `256x512x256` zero-bias shape. It proved PyCUDA can compile C++ WMMA
  sources when `SourceModule(..., no_extern_c=True)` is used, and correctness
  passed. Performance did not qualify: the target improved slightly, but the
  focused score stayed negative. A one-warp-per-16x16 WMMA tile is not enough;
  a real generated path needs multi-warp tiles, shared-memory staging,
  epilogue scheduling, and autotuned variants.
- `552c315` routed square matmul through `cublasGemmEx` with
  `CUBLAS_COMPUTE_32F_FAST_TF32`. The API worked, but it produced no
  saved-best focused improvements and regressed seven focused rows. For this
  benchmark, the existing legacy `cublasSgemm_v2` plus TF32 math mode remains
  the better vendor path; future square-GEMM work needs plan/autotune evidence
  before replacing it.
- `1bf272a` replaced the kept exact 64x64 shared-memory tiled kernel with a
  direct global-load dot-product kernel. It passed correctness but regressed
  `matmul_64x64_x_64x64` on confirmation. This reinforces the FalconGEMM and
  multi-versioning lesson: even tiny GEMM variants need generated/selected
  strategies and repeated measurements, not a single plausible memory-hierarchy
  swap.
- `2211a84` added an exact zero-bias `128x256x128` fused matmul+ReLU kernel
  with no boundary checks, no bias load, and no runtime shape arguments. It
  repeatedly improved both 128 fused-mm rows, but failed the focused gate twice
  because unrelated focused rows regressed. This reinforces the CODA/CUDA Tile
  lesson: NovaX's fused-mm edge probably needs a Tensor Core epilogue or
  autotuned generated tile family, not another one-off SIMT 16x16 tile.

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

- PyGraph reinforces that the durable CUDA Graph frontier is compiler/runtime
  integration. NovaX's graph wins should move toward shape-guarded lowering and
  capture-safe memory planning instead of Python replay conveniences alone.
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
- `53d31b8` removed exact-tile boundary predicates from the naive fused-mm
  kernel. It did not produce a durable target win. The likely bottleneck is
  still arithmetic throughput/memory scheduling versus vendor Tensor Core paths,
  not the boundary guards in the current benchmark shapes.
- `4a31903` retested direct fused-expression building as a Python front-end
  optimization. The result was directionally good for `fusion_chain5`, but not
  strong enough to pass the focused gate. This supports the AutoKernel/FuseFlow
  lesson: keep ranking bottlenecks and optimize lower-level schedules rather
  than spending many turns on Python-only expression-builder shortcuts.
- `d610a5c` specialized exact 64x64 matmul and qualified twice. This is a
  useful counterexample to failed fused-mm exact tiling: the successful case
  only touched one benchmark shape and removed both bounds checks and runtime
  shape parameters from a small hand-written GEMM path.
- `c7f15d8` tried to shave Python-side dispatch overhead from the kept 64x64
  path by checking it before the cuBLAS gate. The target row improved twice,
  but the focused suite failed twice on unrelated noise. This reinforces that
  small launch-dispatch cleanups are below the noise floor unless the full gate
  stays green.
- `2a90d48` cached finalized fused launch kernels to avoid rebuilding CUDA
  source strings on repeated fused expressions. It qualified once, then failed
  confirmation and tiebreaker. This supports the KernelAgent/AutoKernel lesson:
  optimization turns should prioritize profiled bottlenecks over plausible
  Python-cache edits once timings are near the noise floor.
- `5eb4d27` changed fused kernels containing `expf`/`tanhf` to use 256-thread
  blocks. It failed twice and regressed fusion-chain guardrails. This supports
  the TritonForge/KernelAgent lesson: block-size scheduling should be driven by
  profiler counters or an explicit autotune sweep, not a static guess.
- `54cd5f4` row-coarsened the exact 64x64 matmul kernel from 256 threads per
  block to 128 threads per block with each thread producing two output rows.
  Correctness passed, but the target `matmul_64x64_x_64x64` regressed by 1.62x
  versus the saved baseline. This supports the CudaForge/CUDA Agent lesson:
  shape-specific small-GEMM variants need profiler feedback or autotuned search,
  because plausible occupancy reductions can lose to the simpler kept tile.
- `e5e58ed` lowered `a*b + c` fused expression trees to explicit CUDA `fmaf`.
  Correctness passed, but `fusion_chain5_n1000000` regressed by 1.44x versus
  the saved baseline and the focused gate failed. This supports the
  Astra/Nautilus lesson: local instruction substitutions are too shallow when
  the profitable frontier is fusion planning, memory traffic, and schedule
  selection.
- `7b68fc5` narrowed the previous fused-mm cuBLAS idea to the exact zero-bias
  `256x512x256` benchmark shape, using TF32 cuBLAS for GEMM followed by an
  in-place ReLU epilogue. It qualified twice with no focused regressions and
  improved `fused_mm_linear_256_512_256` by 1.22x on the primary run and 1.51x
  on confirmation. This supports the CODA/NVIDIA epilogue lesson with a caveat:
  even a separate epilogue can win when the shape gate is narrow and the naive
  FP32 tile is the real bottleneck.
- `277a76f` vectorized the kept exact fused-mm in-place ReLU epilogue with
  `float4` loads/stores. Correctness passed, but `fused_mm_naive_256_512_256`
  regressed by 1.13x and the focused gate failed. This supports the CODA and
  hardware-feedback lesson: after the GEMM route is fixed, the epilogue launch
  dominates enough that local vectorization can be neutral or harmful.
- `2d4e508` broadened the exact zero-bias fused-mm cuBLAS path to the smaller
  `128x256x128` shape. Correctness passed, but both smaller fused-mm rows
  regressed, with `fused_mm_linear_128_256_128` at 1.38x the saved best. This
  supports the multi-versioning lesson: a vendor-library route that wins at one
  shape can lose at the adjacent smaller shape, so the gate must stay exact.
- `5dd0e23` kept the scalar exact fused-mm ReLU epilogue but removed its
  runtime `total` argument and bounds check. Correctness passed, but the run
  had zero focused improvements and failed the gate. This reinforces the
  launch-overhead lesson: once the epilogue is a separate tiny launch, trimming
  one branch inside the kernel is below the meaningful optimization surface.
- `81fe1de` used 256-thread blocks only for ReLU-only fused expressions. The
  run did not produce enough fusion-chain gain to overcome focused-suite noise
  and regressed `matmul_64x64_x_64x64`. This reinforces the profiling lesson:
  block-size tuning should be explicit autotuning with repeated measurements,
  not another static rule.
- `a4120d8` coarsened large same-size fused elementwise kernels so each thread
  produced four adjacent outputs. Correctness passed, but both fusion-chain
  rows regressed and `fusion_chain3_n1000000` lost to PyTorch. This supports
  the FuseFlow/Nautilus lesson: profitable fusion is a scheduling/dataflow
  problem, not a simple thread-count reduction.
- `1765103` cached large same-size fused-kernel inputs into local scalars to
  avoid repeated global-load syntax in expressions like
  `sigmoid(relu(a*b+c)*a)`. Correctness passed, but `fusion_chain5_n1000000`
  regressed hard and seven focused rows failed. This reinforces the
  compiler/autotuning lesson: local source rewrites can increase register
  pressure or block compiler choices even when they look like obvious CSE.
- `ff3a8ac` validated that exact cuBLASLt ReLU epilogues can run correctly in
  NovaX, but its confirmation benchmark failed despite a primary qualification.
  The new lesson is not "avoid cuBLASLt"; it is "do not put cuBLASLt descriptor
  and heuristic plumbing in the Python hot path and expect stable focused-suite
  wins."
- `7c5ff68` validated the first local WMMA compile path but also showed that
  "uses Tensor Cores" is not the same as "is a competitive GEMM kernel." Future
  generated fused-mm work should start from CUTLASS-style tiling or an autotune
  search space, not a single static warp tile.
- `91b30af` skipped redundant same-size broadcast expression rewriting in
  `launch_fused`. Correctness passed, but the intended fusion rows did not
  improve versus the saved best and the focused gate failed. This reinforces
  the Hybrid JIT-CUDA Graph and KernelBlaster lesson: NovaX should stop
  spending main-loop budget on Python string/dispatch micro-cleanups unless
  profiling shows they dominate a focused target.

## Things Not To Repeat Blindly

- Broad eager elementwise micro-optimizations without a focused-path win.
- General rectangular cuBLAS routing without shape gates.
- cuBLAS state micro-caching unless the matmul or fused-mm targets themselves
  improve.
- Static fused-mm tile changes without profiling or an autotune search.
- Exact-tile fused-mm variants that only remove boundary checks.
- GEMM plus a separate epilogue launch as a substitute for true epilogue fusion.
- Fast-math substitutions without repeatable target-case wins.
- Pointer qualifier or signature-only fused-kernel tweaks.
- Zero-bias special cases inside the naive fused-mm tile.
- Softmax fast-math changes that do not improve the softmax benchmark itself.
- GPU backward rewrites that touch unrelated eager/autograd behavior.
- CUDA graph replay micro-tweaks unless captured inference itself improves.
- Direct fused-expression builder shortcuts unless the full focused gate stays
  green across confirmation runs.
- Shape-specific dispatch micro-tweaks unless they qualify, even when the target
  row itself improves.
- Fused launch-source caching without profiling evidence and repeatable focused
  qualification.
- Static fused-transcendental block-size changes without profiler evidence or
  an autotune sweep.
- Row-coarsened exact 64x64 matmul unless profiler counters show the kept
  16x16 one-output-per-thread tile is occupancy- or synchronization-limited.
- Explicit `fmaf` expression lowering for fusion chains; NVCC likely already
  contracts the multiply-add where profitable.
- Vectorizing the separate ReLU epilogue on the exact 256 fused-mm path; the
  scalar epilogue was faster under the focused gate.
- Broadening the zero-bias fused-mm cuBLAS route to `128x256x128`; the original
  single-kernel fused-mm path is faster for that smaller shape.
- Removing bounds checks from the separate exact fused-mm ReLU epilogue; it did
  not improve the focused suite.
- Static 256-thread block selection for ReLU-only fused expressions without an
  autotune loop or profiler evidence.
- Coarsening fused elementwise kernels by having each thread compute four
  outputs without a scheduler/autotune signal.
- Manual local-input caching in fused elementwise source strings without
  profiler evidence; it regressed the exact chain it was meant to help.
- Python/ctypes cuBLASLt epilogue setup in the hot fused-mm path without a
  persistent lower-level plan cache or generated kernel.
- One-warp-per-output-tile WMMA fused-mm kernels without shared-memory staging,
  multi-warp blocking, or autotuned tile selection.
- Replacing the kept square TF32 `cublasSgemm_v2` path with `cublasGemmEx`
  without measured plan selection; the direct swap regressed the focused suite.
- Direct global-memory exact 64x64 matmul as a replacement for the kept
  shared-memory exact64 kernel; the confirmation run regressed the target row.
- Exact repeated-inference rectangular cuBLAS routing in the current Python
  launcher form; it improved captured inference once but regressed seven
  focused rows, including square matmul.
- Exact zero-bias `128x256x128` fused-mm through another static 16x16 SIMT
  tile; it improved target rows but still failed focused confirmation.
- Same-size fused broadcast-rewrite skipping as a standalone Python-side
  cleanup; it produced no target fused-path win under the focused gate.
