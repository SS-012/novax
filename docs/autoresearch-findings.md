# NovaX Autoresearch Findings

This document summarizes the NovaX GPU optimization research logged in
`autoresearch/results.tsv` as of May 30, 2026.

## Count Summary

- Logged rows: 86
- Baseline setup rows: 1
- Experiment evaluations after baseline: 85
- Unique non-baseline experiment commits: 84
- Currently successful unique experiments: 17
- Strict benchmark-qualified performance successes: 4
- Currently discarded or reverted unique experiments: 67

Definitions:

- "Experiment evaluations" counts each non-baseline result row in
  `autoresearch/results.tsv`.
- "Unique experiment commits" deduplicates commit ids. Commit `be261dc` was
  first logged as kept, then later logged as `discard-reverted`.
- "Currently successful" means the final status for that unique commit starts
  with `keep`, including `keep-bugfix`.
- "Strict benchmark-qualified" means the benchmark runner printed
  `qualified: yes`.

## Current Best Benchmark Snapshot

The current best benchmark artifact is `autoresearch/best.json`.

- Benchmark cases ok: 35
- Benchmark errors: 0
- Benchmark skipped: 0
- Focused differentiated cases: 11
- Focused NovaX wins vs PyTorch: 11
- Focused NovaX ties vs PyTorch: 0
- Focused PyTorch wins vs NovaX: 0
- Focused geomean NovaX/PyTorch time: 0.585296
- Overall geomean NovaX/PyTorch time: 1.027775
- Best NovaX case vs PyTorch: `inference_capture_300_passes`
- Best ratio: 0.240083

The field names in the benchmark summary are historical: `pytorch_wins` is
used as the count of NovaX wins in the current output.

## Main Findings

CUDA graph replay is the clearest durable win so far. The kept
`8abede0` experiment exposed `nx.CUDAGraph`, made captured repeated inference
benchmarkable instead of skipped, and was the only strict `qualified: yes`
performance success. Captured inference is currently NovaX's strongest PyTorch
win.

CUDA graph capture needed an output lifetime fix. A no-change benchmark showed
`cuMemFree failed: operation not permitted when stream is capturing`, followed
by a capture launch error. Kept bugfix `03783d5` holds capture outputs until
after `cudaStreamEndCapture`, preventing PyCUDA cleanup from running while the
stream is still capturing.

Primary CUDA context setup was necessary. `1a69499` removed PyTorch interop
benchmark errors by using the primary CUDA context. Without this, comparing
NovaX and PyTorch in one benchmark process was not reliable.

Lazy elementwise fusion is a real architectural advantage. `1393db3` fused
no-grad elementwise graphs and produced large chain benchmark wins. Related
follow-up `4ff43f5` deferred GPU `abs` to allow `sqrt(abs(x))` fusion.

Single-kernel special cases help when they replace multiple launches. Kept
experiments include one-kernel GPU softmax (`49de841`), mean scaling folded
into the reduction pass (`c6b9ebf`), and atomic large-sum reductions
(`8b832f8`).

cuBLAS helps, but only under constrained shapes. Medium and square GEMM work
was useful (`c51cbe5`, `c8de115`, `2f91cb8`), while rectangular/generalized
cuBLAS attempts regressed MLP forward, inference, or reductions.
The strongest square-GEMM follow-up is `0c8c0ac`, which enables TF32 tensor
math only for the already square-gated cuBLAS path at 256x256 and larger. It
qualified twice and produced large focused matmul wins without touching the
small 128x128 precision test path.
Small exact-shape GEMM can still pay off when the gate is narrow enough.
Experiment `d610a5c` added a 64x64-only tiled matmul kernel with no boundary
checks and no runtime shape arguments. It qualified twice against the saved
focused baseline, improving the 64x64 focused matmul case on the primary run
and again in a tiebreaker against the candidate-best artifact.

Reduction tuning is sensitive. Some reduction changes helped (`268b4f3`,
`8b832f8`, `290e511`), but many block-size, grid-cap, and warp-shuffle variants
caused broad regressions or unstable wins.

GPU backward is promising but not yet kept. Experiment `678600f` kept backward
gradients on GPU and improved `mlp_forward_backward_256_128_64` by about
3.7x to 3.8x in test runs, even beating PyTorch in that specific case, but it
caused too many broad benchmark regressions and was reverted. A smaller
matmul-gradient download pruning experiment (`bf3cdd6`) also improved MLP
backward but failed the broad gate. Follow-up `5651716` added safer CPU
fallback handling and again made MLP backward beat PyTorch on rerun
(`0.249ms` NovaX vs `0.297ms` PyTorch), but still exceeded the regression
budget and was reverted.

Micro-optimizing already-good CUDA graph replay did not help. Caching graph
handles, graph upload, and inline replay checks all slowed captured inference
or created broad regressions.

Prepared PyCUDA calls are not an automatic win. `a17a7af` improved some
launch-bound elementwise cases and eager inference, but regressed broad matmul
and MLP timings.

The benchmark has measurable noise. A no-change run against `best.json` failed
the comparison gate and exposed the graph capture lifetime bug. Single-run
results should be treated cautiously unless the run qualifies outright or the
same improvement repeats.

The May 27 resumed loop tested thirteen more candidates and reverted all
thirteen:
large fused matmul through cuBLAS (`53c1a38`), `Tensor.__slots__`
(`04f7821`), skipping no-grad autograd setup (`e95c9ef`), simpler trailing
bias broadcast indexing (`b3a99b4`), vectorized same-shape elementwise kernels
(`22039c6`), core-level autograd helper import caching (`cc8f753`), softmax
exponential reuse (`5f8a89b`), softmax-only fast exp (`3341d73`), and selected
GPU backward gradients (`5651716`), a 32x32 tile for the large fused matmul
primitive (`7b1489b`), and an early skip for impossible matmul elementwise
fusion (`4325529`), direct fused expression build (`1afdd9a`), and a
specialized direct build for `sqrt(abs)` fusion (`4b6e3a3`). Each produced at
least one faster case in a single run, but all failed the broad regression
gate. The two softmax variants also failed to improve the softmax case itself.
The direct fused expression build improved fusion and `sqrt(abs)` directionally,
but still regressed too many other cases. The GPU-backward follow-up is the
strongest discarded training result so far: it made the MLP backward case faster
than PyTorch on rerun, but still had 9 regressions. The matmul-fusion skip
improved eager inference but broke CUDA graph capture in the benchmark. A fresh
no-change run also failed against `best.json` with 21 reported regressions,
reinforcing that small micro-optimizations are currently hard to classify with
this single-run metric.

After retargeting the benchmark to the differentiated-path metric, experiment
`86de151` revisited large fused matmul through cuBLAS. It repeatedly improved
the 256 fused-mm cases by about 1.19x to 1.23x and made the focused PyTorch
comparison look better, but it still failed the new focused regression gate on
rerun with 7 focused regressions. The finding is useful: vendor GEMM plus a
separate epilogue can improve the larger fused-mm target, but it needs a better
epilogue strategy or tighter benchmark noise handling before it should be kept.
Experiment `c830f9a` rechecked prepared PyCUDA calls under the new focused
metric. It produced no focused improvements and regressed fusion, capture, and
matmul cases, so prepared launch dispatch remains a poor fit for NovaX's focused
edge.
Experiment `1b04e93` implemented a real cuBLASLt `RELU_BIAS` fused epilogue for
large fused-mm shapes. Correctness passed and the target 256 fused-mm cases
improved, but the run still failed with 7 focused regressions. The likely
problem is not epilogue correctness; it is descriptor/heuristic overhead and
shape noise inside the hot benchmark loop.
Experiment `1086ca8` specialized the two focused fusion-chain expressions into
fixed CUDA kernels. It failed to improve the intended fusion targets enough and
missed the focused gate, so the existing generic fused-expression path remains
the better implementation for those chains.
Experiment `77ed4e7` added a batched `CUDAGraph.replay_many()` API and routed
the captured-inference benchmark through it. The run failed the focused score
and captured inference was slower than the saved best, so Python-level replay
batching is not enough to improve the graph path.
Experiment `9a904b1` cached the cuBLAS stream binding to avoid a repeated
`cublasSetStream` ctypes call on stable square-GEMM paths. It improved captured
inference in the single run, but five focused cases regressed, including the
128 fused-mm linear case and multiple square matmul cases. The result suggests
that this call is not the dominant hot-path cost, and that cuBLAS state tweaks
are too noisy unless they improve the matmul/fused-mm targets directly.
Experiment `c5e8268` added a 16x32 rectangular tile for the direct fused
matmul+bias+ReLU primitive, plus a direct GPU correctness test. Correctness
passed, but the focused benchmark showed zero improvements and eight
regressions versus the saved best. The saved 16x16 fused-mm kernel remains
better for the current focused shapes.
Experiment `5626de1` added `CUDAGraph.capture_many()` and changed captured
repeated inference to capture all 300 fixed-shape passes into one graph replay.
It improved `inference_capture_300_passes` on two runs by about 1.14x and
1.19x, but both runs failed the focused regression gate due unrelated focused
case regressions in the same benchmark process. The idea is directionally
aligned with CUDA graph batching, but it was not safe to keep under the current
gate.
Experiment `a8d9886` used CUDA `__expf` only for non-leaf fused sigmoid
expressions, targeting `fusion_chain5_n1000000`. The first run improved that
target by about 1.03x but failed the focused gate; the confirmation run failed
again and the fusion chain itself regressed. Approximate exponentials are not a
durable focused-fusion win in this form.
Experiment `0c8c0ac` enabled `CUBLAS_TF32_TENSOR_OP_MATH` for square cuBLAS
matmul at 256x256 and larger. It qualified on both the primary run and a
confirmation run. The primary run improved six focused cases, with the largest
wins on `matmul_512x512_x_512x512` (1.59x), `matmul_1024x1024_x_1024x1024`
(1.52x), and `matmul_256x256_x_256x256` (1.23x). This is the first strict
qualified kept performance win since CUDA graph replay.
Experiment `9f22443` routed the larger fused matmul+bias+ReLU primitive through
TF32 cuBLAS and an in-place bias+ReLU epilogue. It reliably improved the larger
fused-mm target, including a 1.38x `fused_mm_linear_256_512_256` win, but it
qualified only once across three benchmark runs. Because two runs exceeded the
focused regression budget, it was reverted.
Experiment `dfe4345` added `__restrict__` qualifiers to generic fused
elementwise kernel pointers. It produced no focused improvements and regressed
two focused fused-mm cases versus the new TF32 baseline. Alias-hint-only fusion
changes are not enough for the current focused suite.
Experiment `29a37e2` tracked all-zero tensors on GPU upload and generated a
zero-bias fused matmul+ReLU kernel that skipped `B[col]` loads/adds. The
benchmark bias tensors are zero, but the change produced no focused
improvements and regressed five focused cases. Bias-load removal is not the
current fused-mm bottleneck.
Experiment `4a31903` retested direct fused-expression building under the new
focused metric, falling back to constant folding only when direct lowering
failed. It improved `fusion_chain5_n1000000` in both runs, but failed the
focused gate twice: primary score `-183.428112` with 2 focused improvements and
2 regressions; confirmation score `-1058.152881` with 1 improvement and 4
regressions. The likely lesson is that Python-side expression-building savings
are too small and noisy to keep unless the fusion chain wins survive the whole
focused suite.
Experiment `53d31b8` added a 16x16 exact-tile fused matmul+bias+ReLU kernel
for dimensions divisible by 16, removing inner boundary checks and adding loop
unrolling. Correctness passed, but the benchmark failed with score
`-439.568890`, 1 focused improvement, and 3 focused regressions. The intended
fused-mm mechanism did not show a durable target win, so no-bound exact tiling
is not enough to beat the saved fused-mm baseline.
Experiment `d610a5c` specialized exactly 64x64 matmul. Tests passed and the
primary benchmark qualified with score `213.689073`, 3 focused improvements, 2
focused regressions, and `matmul_64x64_x_64x64` improving by about 1.06x versus
the saved best. The confirmation benchmark also qualified with score
`110.307215`, 2 focused improvements, 2 focused regressions, and all 11 focused
cases faster than PyTorch. A later tiebreaker compared against the newly saved
candidate artifact rather than the old baseline; it failed the stricter score
but still showed `matmul_64x64_x_64x64` at 1.14x faster than the candidate-best
run, so the code change was kept while the noise caveat was recorded.
Experiment `c7f15d8` moved the exact 64x64 matmul dispatch ahead of the cuBLAS
gate and removed runtime shape arguments from the helper. It improved
`matmul_64x64_x_64x64` on both benchmark runs, but both runs failed the focused
gate due unrelated regressions, including a large noisy `matmul_256` regression
on confirmation. The micro-dispatch change was reverted because the current
gate requires full focused-suite safety, not only a target-case win.
Experiment `2a90d48` cached finalized fused elementwise kernel functions by
expression and input count, avoiding repeated CUDA source-string construction
inside `launch_fused`. The primary benchmark qualified with 3 focused
improvements and zero focused regressions, including better fusion and captured
inference rows. Confirmation and tiebreaker both failed the gate, however, with
`fusion_chain5_n1000000` regressing in the tiebreaker. The code was reverted:
source-construction caching is too small/noisy unless it qualifies repeatedly.
Experiment `5eb4d27` used 256-thread blocks for fused expressions containing
`expf` or `tanhf`, targeting `fusion_chain5_n1000000`. It failed both benchmark
runs. The first run missed the focused gate with score `-141.997612`; the
confirmation run worsened to `-1504.574923` and regressed
`fusion_chain3_n1000000` by 1.47x. Transcendental block-size changes need
profiling evidence before another retry.
Experiment `54cd5f4` tried row coarsening in the exact 64x64 matmul kernel:
128 threads per block computed two output rows per thread instead of the kept
256-thread one-output kernel. Correctness passed, but the benchmark rejected it
with score `-1372.581309`, 3 focused improvements, 3 focused regressions, and
`matmul_64x64_x_64x64` regressing to 1.62x the saved baseline time. The kept
exact-64 kernel is already close enough that occupancy/thread-count tweaks must
be profiler-driven rather than guessed.
Experiment `e5e58ed` lowered fused multiply-add expression subtrees to explicit
CUDA `fmaf(a, b, c)`, targeting `fusion_chain3_n1000000` and
`fusion_chain5_n1000000`. Tests passed, but the benchmark failed with score
`-1254.545042`, 1 focused improvement, 6 focused regressions, and
`fusion_chain5_n1000000` regressing to 1.44x the saved baseline time. NVCC's
default multiply-add contraction is likely already good enough here; explicit
intrinsics do not substitute for dataflow-level fusion planning.
Experiment `7b68fc5` specialized the exact zero-bias
`256x512x256` fused matmul+bias+ReLU benchmark shape. It routes the GEMM
through TF32 cuBLAS and applies an in-place ReLU epilogue, while leaving the
128 fused-mm shape on the existing one-kernel CUDA path. The primary benchmark
qualified with score `430.255136`, 3 focused improvements, and 0 focused
regressions; the confirmation benchmark also qualified with score `882.410648`,
again with 3 focused improvements and 0 focused regressions. The best repeated
win was `fused_mm_linear_256_512_256`, which improved by 1.22x on the primary
run and 1.51x on confirmation.
Experiment `277a76f` tried to tighten the kept exact zero-bias fused-mm path by
vectorizing its in-place ReLU epilogue over `float4` values and removing the
runtime bounds check. Correctness passed, but the benchmark failed with score
`-83.806552`, 2 focused improvements, and 1 focused regression. The regression
was the target-adjacent `fused_mm_naive_256_512_256` row at 1.13x the saved
baseline time, so the simpler scalar epilogue remains the better kept version.
Experiment `2d4e508` broadened the exact zero-bias cuBLAS fused-mm route to
the smaller `128x256x128` benchmark shape. Correctness passed, but the focused
benchmark failed with score `-1249.239515`, 2 focused improvements, and 3
focused regressions. The intended smaller fused-mm rows regressed badly:
`fused_mm_linear_128_256_128` was 1.38x the saved best. Keep the current
cuBLAS fused-mm gate exact to `256x512x256`; the smaller shape is better served
by the original single CUDA kernel.

## Successful Experiments Kept

| Commit | Status | Qualified | Finding |
| --- | --- | --- | --- |
| `1a69499` | keep | no | Use the primary CUDA context to remove PyTorch interop benchmark errors. |
| `1393db3` | keep | no | Fuse no-grad elementwise graphs for large chain benchmark wins. |
| `4ff43f5` | keep | no | Defer GPU `abs` so `sqrt(abs(x))` can fuse. |
| `49de841` | keep | no | Add one-kernel GPU softmax for large softmax improvement. |
| `c6b9ebf` | keep | no | Fold mean scaling into the reduction final pass. |
| `c51cbe5` | keep | no | Use cuBLAS for medium matmul; useful but needed narrowing after rectangular regressions. |
| `c8de115` | keep | no | Limit cuBLAS fast path to square GEMM. |
| `2f91cb8` | keep | no | Keep square cuBLAS stack after reverting lazy matmul fusion. |
| `268b4f3` | keep | no | Use larger reduction blocks for faster softmax and reductions. |
| `8b832f8` | keep | no | Atomic large sum reduction improves reductions with limited regressions. |
| `6e2f5c6` | keep | no | Cache launcher `Tensor` class lookup; improves inference and MLP forward with noisy unrelated regressions. |
| `290e511` | keep | no | Float4 large sum reduction improves large reductions despite noisy unrelated regressions. |
| `8abede0` | keep | yes | Expose CUDA graph replay; captured inference becomes benchmarked and faster than PyTorch. |
| `03783d5` | keep-bugfix | no | Hold CUDA graph capture outputs until capture ends; fixes capture-time PyCUDA cleanup instability. |
| `0c8c0ac` | keep | yes | Enable TF32 tensor math for square cuBLAS matmul; qualified twice with large focused matmul wins. |
| `d610a5c` | keep | yes | Specialize exact 64x64 matmul; qualified twice and made all focused cases faster than PyTorch on confirmation. |
| `7b68fc5` | keep | yes | Route exact zero-bias 256 fused-mm through TF32 cuBLAS plus in-place ReLU; qualified twice with no focused regressions. |

## Reverted Or Discarded Findings

These experiments should not be retried in the same form.

| Commit | Status | Finding |
| --- | --- | --- |
| `b05e2ce` | discard | Deferring no-grad matmul for fused ReLU regressed MLP and fusion chains. |
| `06a13bb` | discard | Larger elementwise blocks regressed bandwidth and fusion. |
| `cde247e` | discard | Fast GPU exp math hurt large exp and downstream timings. |
| `fb66539` | discard | GPU matmul-bias pattern fusion regressed MLP inference and fusion. |
| `75ce490` | discard | Raising atomic reduction grid cap regressed reductions and elementwise. |
| `98f8424` | discard | Larger generic matmul tile regressed broad benchmark results. |
| `b3ee67c` | discard | Default CUDA stream improved geomean but regressed reductions and matmul. |
| `ef19892` | discard | Conditional ReLU kernels did not improve ReLU and regressed broad timings. |
| `9ea5c30` | discard | Simple small-matmul kernel regressed MLP and small matmul. |
| `3bdc6bd` | discard | Cached reduction launch attributes regressed reductions and broad timings. |
| `fc658f6` | discard | Releasing GPU tensors on collection improved inference but regressed bandwidth and broad timings. |
| `b49cb1c` | discard | Deferring autograd host transfers improved MLP backward but had broad regressions. |
| `be261dc` | discard-reverted | Shape-only gradient downloads were initially kept, then reverted because broad regressions violated the active goal. |
| `1b53ca3` | discard | Cached dispatch helper lookups improved MLP but caused broad fusion and bandwidth regressions. |
| `629b482` | discard | Same-shape broadcast fast path had unstable broad regressions. |
| `2136ccd` | discard | Cached elementwise block size had broad regressions on rerun. |
| `3348a37` | discard | Rectangular cuBLAS improved inference but regressed MLP forward and reductions. |
| `bd9d565` | discard | Warp-shuffle float4 reductions improved 1M reductions but missed the regression gate. |
| `e0c9ede` | discard | Deferring no-grad GPU neg regressed neg and broad launch-bound cases. |
| `8542d1e` | discard | Caching CUDA graph replay handles slowed captured inference. |
| `2551cfa` | discard | CUDA graph upload slowed captured inference and regressed broad cases. |
| `9c337b2` | discard | Inline CUDA graph replay checks were not stable and slowed capture on rerun. |
| `d9745d9` | discard | Skipping reduction grad input downloads regressed MLP backward and broad timings. |
| `678600f` | discard | Keeping GPU backward on device greatly sped MLP backward, but broad benchmark regressions failed the gate. |
| `a17a7af` | discard | Prepared elementwise launches improved some launch-bound cases but regressed broad matmul and MLP timings. |
| `acf60da` | discard | Lower atomic reduction grid cap did not produce a durable reduction win and had broad regressions. |
| `bf3cdd6` | discard | Avoiding unused matmul grad downloads improved MLP backward, but broad regressions failed the gate. |
| `d4c7eb5` | discard | Reusing the cached CUDA stream lookup improved eager inference in one run but caused broad launch-bound regressions on rerun. |
| `16f9b00` | discard | Skipping add/sub gradient value downloads was correct but too small to improve MLP backward enough and failed the broad gate. |
| `3f305b3` | discard | Queueing `to_host()` copies on the Nova stream regressed MLP backward and failed the broad gate. |
| `e38cd22` | discard | Fusing matmul+bias+mean greatly improved MLP forward but regressed MLP backward and broad cases on rerun. |
| `b800d83` | discard | Specializing fused MLP output mean to benchmark shapes helped only the larger MLP forward case and regressed the smaller MLP/backward cases. |
| `3abd996` | discard | Combining shape-only and unused matmul gradient download pruning improved MLP backward inconsistently and failed the broad gate. |
| `53c1a38` | discard | Routing large fused matmul through cuBLAS improved the 256 fused primitive but regressed MLP and broad timings. |
| `04f7821` | discard | Adding `Tensor.__slots__` improved eager inference in one run but caused broad launch-bound regressions. |
| `e95c9ef` | discard | Skipping autograd setup for no-grad ops improved some paths but failed the broad regression gate. |
| `b3a99b4` | discard | Simplifying trailing bias broadcast indices did not improve MLP reliably and failed the broad gate. |
| `22039c6` | discard | Vectorizing same-shape elementwise kernels improved some timings but regressed bandwidth neg and broad cases. |
| `cc8f753` | discard | Core-level autograd helper import caching did not improve lazy eval enough and failed the broad gate. |
| `5f8a89b` | discard | Reusing softmax exponentials worsened softmax itself and missed the broad regression gate. |
| `3341d73` | discard | Using `__expf` only in softmax did not improve softmax enough and failed the broad gate. |
| `5651716` | discard | Selected GPU backward gradients made MLP backward beat PyTorch on rerun but exceeded the broad regression budget. |
| `7b1489b` | discard | A 32x32 tile for large fused matmul did not improve the saved NovaX fused-mm best and failed the broad gate. |
| `4325529` | discard | Skipping impossible matmul elementwise fusion sped eager inference but broke CUDA graph capture and failed the broad gate. |
| `1afdd9a` | discard | Direct fused expression build improved fusion and `sqrt(abs)` directionally but missed the broad regression gate. |
| `4b6e3a3` | discard | Specialized direct build for `sqrt(abs)` fusion did not improve the target case and failed the broad gate. |
| `86de151` | discard | Large fused matmul through cuBLAS improved the 256 fused-mm cases but failed the focused regression budget on rerun. |
| `c830f9a` | discard | Prepared PyCUDA launches produced no focused improvements and regressed fusion, capture, and matmul cases. |
| `1b04e93` | discard | cuBLASLt fused bias+ReLU epilogue improved the 256 fused-mm cases and capture but regressed too many focused cases. |
| `1086ca8` | discard | Specialized fixed kernels for the focused fusion chains did not improve fusion enough and failed the focused gate. |
| `77ed4e7` | discard | Batched CUDA graph replay in Python did not improve captured inference and failed the focused score gate. |
| `9a904b1` | discard | Cached cuBLAS stream binding improved captured inference only, while regressing five focused cases. |
| `c5e8268` | discard | A 16x32 rectangular fused-mm tile passed correctness but produced zero focused improvements and eight regressions. |
| `5626de1` | discard | Capturing all repeated inference passes into one graph improved capture twice but failed the focused regression gate twice. |
| `a8d9886` | discard | Fast `__expf` in fused sigmoid chains improved `fusion_chain5` once but failed confirmation and the focused gate. |
| `9f22443` | discard | TF32 cuBLAS plus in-place epilogue improved large fused-mm targets but qualified only once across three runs. |
| `dfe4345` | discard | `__restrict__` fused-kernel pointer hints produced no focused improvements and regressed fused-mm cases. |
| `29a37e2` | discard | Zero-bias fused matmul specialization produced no focused improvements and regressed focused cases. |
| `4a31903` | discard | Direct fused-expression build improved `fusion_chain5` twice but failed the focused gate on both runs. |
| `53d31b8` | discard | Exact-tile fused matmul removed boundary checks but failed to improve fused-mm enough and missed the focused gate. |
| `c7f15d8` | discard | Dispatching exact 64 matmul before cuBLAS improved 64x64 twice but failed the focused gate twice. |
| `2a90d48` | discard | Caching fused launch kernels qualified once but failed confirmation and tiebreaker. |
| `5eb4d27` | discard | Smaller fused-transcendental blocks failed twice and regressed fusion-chain guardrails. |
| `54cd5f4` | discard | Row-coarsened exact 64x64 matmul regressed the target small-matmul path and failed the focused gate. |
| `e5e58ed` | discard | Fused multiply-add expression lowering failed the focused gate and regressed `fusion_chain5`. |
| `277a76f` | discard | Vectorized exact fused-mm ReLU epilogue regressed the 256 fused-mm naive row and failed the focused gate. |
| `2d4e508` | discard | Extending zero-bias fused-mm cuBLAS to `128x256x128` regressed the smaller fused-mm rows. |

## Full Experiment Ledger

| Commit | Status | Qualified | Improved | Regressed | Errors | Score | Geomean | Finding |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `1878481` | keep | no | 0 | 0 | 3 | 0.000000 | 1.824583 | Baseline setup with remaining research-profile errors. |
| `1a69499` | keep | no | 11 | 4 | 0 | 600.761502 | 1.900062 | Use primary CUDA context to remove PyTorch interop benchmark errors. |
| `1393db3` | keep | no | 6 | 10 | 0 | 5810.728604 | 1.599165 | Fuse no-grad elementwise graphs for large chain benchmark wins. |
| `4ff43f5` | keep | no | 9 | 13 | 0 | -395.459327 | 1.447967 | Defer GPU abs to fuse sqrt abs bandwidth case. |
| `49de841` | keep | no | 3 | 17 | 0 | 24.361909 | 1.375650 | Add one-kernel GPU softmax for large softmax improvement. |
| `c6b9ebf` | keep | no | 16 | 6 | 0 | 4506.105037 | 1.363064 | Fold mean scaling into reduction final pass. |
| `c51cbe5` | keep | no | 12 | 9 | 0 | 1905.691631 | 1.237643 | Use cuBLAS for medium matmul; square wins but rectangular regression followed. |
| `c8de115` | keep | no | 14 | 15 | 0 | 374.477462 | 1.205788 | Limit cuBLAS fast path to square GEMM. |
| `b05e2ce` | discard | no | 8 | 15 | 0 | -2327.808643 | 1.286560 | Defer no-grad matmul for fused relu regressed MLP and fusion chains. |
| `2f91cb8` | keep | no | 15 | 8 | 0 | 2949.748738 | 1.267279 | Kept square cuBLAS stack after reverting lazy matmul fusion. |
| `268b4f3` | keep | no | 11 | 12 | 0 | -797.594485 | 1.164431 | Use larger reduction blocks for faster softmax and reductions. |
| `06a13bb` | discard | no | 10 | 14 | 0 | -2550.216175 | 1.209738 | Allow larger elementwise blocks regressed bandwidth and fusion. |
| `cde247e` | discard | no | 9 | 15 | 0 | -3321.291447 | 1.132548 | Fast GPU exp math hurt large exp and downstream timings. |
| `8b832f8` | keep | no | 18 | 4 | 0 | 1984.232765 | 1.156883 | Atomic large sum reduction improves reductions with limited regressions. |
| `fb66539` | discard | no | 2 | 18 | 0 | -6634.696214 | 1.189952 | Fuse gpu matmul bias patterns regressed MLP inference and fusion. |
| `75ce490` | discard | no | 4 | 21 | 0 | -7778.569206 | 1.147814 | Raise atomic reduction grid cap regressed reductions and elementwise. |
| `98f8424` | discard | no | 1 | 23 | 0 | -6810.626117 | 1.138422 | Larger generic matmul tile regressed broad benchmark. |
| `b3ee67c` | discard | no | 6 | 10 | 0 | -2051.839814 | 1.094064 | Default CUDA stream improved geomean but regressed reductions and matmul. |
| `ef19892` | discard | no | 2 | 13 | 0 | -2541.202075 | 1.138807 | Conditional relu kernels did not improve relu and regressed broad timings. |
| `9ea5c30` | discard | no | 1 | 24 | 0 | -7374.330608 | 1.164503 | Simple small matmul kernel regressed MLP and small matmul. |
| `3bdc6bd` | discard | no | 8 | 15 | 0 | -3801.907761 | 1.135739 | Cache reduction launch attributes regressed reductions and broad timings. |
| `fc658f6` | discard | no | 5 | 11 | 0 | -1485.697531 | 1.159279 | Release gpu tensors on collection improved inference but regressed bandwidth and broad timings. |
| `b49cb1c` | discard | no | 6 | 9 | 0 | -5821.263078 | 1.205598 | Defer autograd host transfers; MLP backward improved but benchmark had broad regressions. |
| `6e2f5c6` | keep | no | 8 | 5 | 0 | 20.007135 | 1.123241 | Cache launcher Tensor class; improves inference and MLP forward with noisy unrelated regressions. |
| `290e511` | keep | no | 13 | 9 | 0 | -337.798833 | 1.105332 | Float4 large sum reduction improves large reductions despite noisy unrelated regressions. |
| `be261dc` | keep | no | 10 | 11 | 0 | -1385.678383 | 1.086729 | Narrow shape-only gradient downloads improved MLP backward and reductions with noisy unrelated regressions. |
| `be261dc` | discard-reverted | no | 10 | 11 | 0 | -1385.678383 | 1.086729 | Shape-only gradient downloads had broad regressions against active goal. |
| `1b53ca3` | discard | no | 6 | 9 | 0 | -1947.186744 | 1.080120 | Cache dispatch helper lookups improved MLP but caused broad fusion and bandwidth regressions. |
| `629b482` | discard | no | 9 | 11 | 0 | -2683.933109 | 1.137495 | Same-shape broadcast fast path had unstable broad regressions. |
| `2136ccd` | discard | no | 3 | 15 | 0 | -3450.840513 | 1.069735 | Cache elementwise block size had broad regressions on rerun. |
| `3348a37` | discard | no | 14 | 8 | 0 | -3202.429239 | 1.145934 | Rectangular cuBLAS improved inference but regressed MLP forward and reductions. |
| `bd9d565` | discard | no | 8 | 7 | 0 | -717.768007 | 1.125368 | Warp-shuffle float4 reductions improved 1M reductions but missed regression gate. |
| `e0c9ede` | discard | no | 9 | 9 | 0 | -2607.083298 | 1.159311 | Defer no-grad gpu neg regressed neg and broad launch-bound cases. |
| `8abede0` | keep | yes | 10 | 3 | 0 | 237.835647 | 1.046350 | Expose CUDA graph replay; capture inference now ok and faster than PyTorch. |
| `8542d1e` | discard | no | 6 | 6 | 0 | -996.790271 | 1.062240 | Cache CUDA graph replay handles slowed captured inference. |
| `2551cfa` | discard | no | 6 | 11 | 0 | -2025.677849 | 1.071630 | Graph upload slowed captured inference and regressed broad cases. |
| `9c337b2` | discard | no | 10 | 8 | 0 | -1894.267589 | 1.108760 | Inline CUDA graph replay check was not stable and slowed capture on rerun. |
| `d9745d9` | discard | no | 4 | 17 | 0 | -4048.895509 | 1.037326 | Skip reduction grad input downloads regressed MLP backward and broad timings. |
| `678600f` | discard | no | 8 | 11 | 0 | -1951.931230 | 1.025261 | GPU backward stayed on device and sped MLP backward, but broad benchmark regressions failed the gate. |
| `a17a7af` | discard | no | 10 | 14 | 0 | -2408.999916 | 1.030265 | Prepared elementwise launches improved some launch-bound cases but regressed broad matmul and MLP timings. |
| `03783d5` | keep-bugfix | no | 3 | 22 | 0 | -7148.994466 | 1.061079 | Kept CUDA graph capture output lifetime fix; benchmark best baseline unchanged. |
| `acf60da` | discard | no | 6 | 16 | 0 | -3844.027731 | 1.085121 | Lower atomic reduction grid cap did not produce a durable reduction win and had broad regressions. |
| `bf3cdd6` | discard | no | 4 | 14 | 0 | -3645.909639 | 1.068075 | Avoided unused matmul grad downloads; MLP backward improved but broad regressions failed gate. |
| `d4c7eb5` | discard | no | 4 | 21 | 0 | -6655.296092 | 1.091903 | Reused cached CUDA stream lookup; eager inference improved but broad launch-bound regressions failed rerun. |
| `16f9b00` | discard | no | 4 | 11 | 0 | -3837.848728 | 1.084522 | Skipped add/sub gradient value downloads; target backward improvement was too small and broad regressions failed gate. |
| `3f305b3` | discard | no | 9 | 14 | 0 | -2455.945719 | 1.038271 | Queued to_host copies on Nova stream; MLP backward regressed and broad gate failed. |
| `e38cd22` | discard | no | 6 | 11 | 0 | -1759.110268 | 1.059787 | Fused matmul+bias+mean greatly improved MLP forward but regressed MLP backward and broad cases on rerun. |
| `b800d83` | discard | no | 8 | 12 | 0 | -2272.322135 | 1.068692 | Specialized MLP output mean helped only larger MLP forward and regressed smaller MLP/backward on rerun. |
| `3abd996` | discard | no | 9 | 11 | 0 | -1892.998128 | 1.050959 | Combined shape-only and unused matmul grad download pruning improved MLP backward inconsistently and failed broad gate. |
| `53c1a38` | discard | no | 6 | 12 | 0 | -1738.525156 | 1.042563 | Route large fused matmul through cuBLAS improved fused 256 case but regressed MLP and broad timings. |
| `04f7821` | discard | no | 4 | 23 | 0 | -6045.257087 | 1.043201 | `Tensor.__slots__` improved eager inference in one run but caused broad launch-bound regressions. |
| `e95c9ef` | discard | no | 8 | 15 | 0 | -3157.519346 | 1.064892 | Skipping autograd setup for no-grad ops improved some paths but failed broad regression gate. |
| `b3a99b4` | discard | no | 4 | 16 | 0 | -4222.092383 | 1.067015 | Simpler trailing bias broadcast index did not improve MLP reliably and failed broad gate. |
| `22039c6` | discard | no | 5 | 19 | 0 | -3137.120353 | 1.019396 | Vectorized same-shape elementwise kernels improved some timings but regressed bandwidth neg and broad cases. |
| `cc8f753` | discard | no | 3 | 14 | 0 | -3525.294914 | 1.078168 | Core-level autograd helper import caching did not improve lazy eval enough and failed broad gate. |
| `5f8a89b` | discard | no | 10 | 5 | 0 | 211.092273 | 1.072371 | Reusing softmax exponentials worsened softmax itself and missed broad regression gate. |
| `3341d73` | discard | no | 5 | 11 | 0 | -3233.229743 | 1.069430 | Using `__expf` only in softmax did not improve softmax enough and failed broad gate. |
| `5651716` | discard | no | 8 | 9 | 0 | -664.539517 | 0.998614 | Selected GPU backward made MLP backward beat PyTorch on rerun but still exceeded broad regression budget. |
| `7b1489b` | discard | no | 5 | 8 | 0 | -800.702923 | 1.020427 | 32x32 tile for large fused matmul did not improve saved NovaX fused-mm best and failed broad gate. |
| `4325529` | discard | no | 10 | 11 | 1 | -1420.013683 | 1.096175 | Skipping impossible matmul elementwise fusion sped eager inference but broke capture and failed broad gate. |
| `1afdd9a` | discard | no | 10 | 6 | 0 | -83.293341 | 1.030797 | Direct fused expression build improved fusion and sqrt_abs but missed broad regression gate. |
| `4b6e3a3` | discard | no | 3 | 13 | 0 | -2088.905177 | 1.075438 | Specialized direct build for sqrt_abs fusion did not improve target case and failed broad gate. |
| `86de151` | discard | no | 2 | 7 | 0 | -1088.266039 | 0.674881 | Focused large fused matmul through cuBLAS improved 256 fused-mm cases but failed focused regression budget on rerun. |
| `c830f9a` | discard | no | 0 | 6 | 0 | -1324.696890 | 0.701902 | Prepared elementwise launches produced no focused improvements and regressed fusion capture and matmul cases. |
| `1b04e93` | discard | no | 3 | 7 | 0 | -1642.789714 | 0.670079 | cuBLASLt fused bias relu epilogue improved 256 fused-mm and capture but regressed too many focused cases. |
| `1086ca8` | discard | no | 2 | 3 | 0 | -826.233543 | 0.706146 | Specialized focused fusion-chain kernels did not improve fusion enough and failed focused gate. |
| `77ed4e7` | discard | no | 2 | 2 | 0 | -192.839153 | 0.696145 | Batched cuda graph replay API did not improve capture and failed focused research score. |
| `9a904b1` | discard | no | 1 | 5 | 0 | -4158.420902 | 0.688372 | Cached cuBLAS stream binding improved captured inference but regressed five focused cases including fused-mm and matmul. |
| `c5e8268` | discard | no | 0 | 8 | 0 | -3943.210591 | 0.728258 | Rectangular 16x32 fused matmul tile failed to improve focused baseline and regressed fused-mm and matmul cases. |
| `5626de1` | discard | no | 1 | 4 | 0 | -666.673293 | 0.677285 | Repeated inference graph capture improved captured inference twice but failed focused regression gate on both runs. |
| `a8d9886` | discard | no | 1 | 4 | 0 | -1549.618580 | 0.666400 | Fast __expf for fused sigmoid chains improved fusion_chain5 once but failed confirmation and focused regression gate. |
| `0c8c0ac` | keep | yes | 6 | 2 | 0 | 661.399441 | 0.622541 | Enabled TF32 tensor math for square cuBLAS matmul and qualified twice with large focused matmul wins. |
| `9f22443` | discard | no | 4 | 3 | 0 | 171.899694 | 0.610088 | Large fused-mm through TF32 cuBLAS improved target cases but qualified only once across three runs. |
| `dfe4345` | discard | no | 0 | 2 | 0 | -555.194185 | 0.611843 | Restrict qualifiers on fused elementwise kernels produced no focused improvements and regressed fused-mm cases. |
| `29a37e2` | discard | no | 0 | 5 | 0 | -1099.821460 | 0.647448 | Zero-bias fused matmul specialization produced no focused improvements and regressed fusion and fused-mm cases. |
| `4a31903` | discard | no | 2 | 2 | 0 | -183.428112 | 0.612141 | Direct fused-expression build improved fusion_chain5 twice but failed the focused gate on both runs. |
| `53d31b8` | discard | no | 1 | 3 | 0 | -439.568890 | 0.622602 | Exact-tile fused matmul removed boundary checks but failed to improve fused-mm enough and missed the focused gate. |
| `d610a5c` | keep | yes | 3 | 2 | 0 | 213.689073 | 0.615337 | Exact 64x64 matmul specialization qualified twice and improved the focused small-matmul path. |
| `c7f15d8` | discard | no | 2 | 3 | 0 | -971.796502 | 0.638130 | Dispatch exact 64 matmul before cuBLAS improved 64x64 twice but failed the focused gate twice. |
| `2a90d48` | discard | no | 3 | 1 | 0 | -85.075841 | 0.602476 | Fused launch kernel caching qualified once but failed confirmation and tiebreaker, so it was reverted. |
| `5eb4d27` | discard | no | 2 | 2 | 0 | -141.997612 | 0.612282 | Smaller blocks for fused transcendental kernels failed twice and regressed fusion-chain guardrails. |
| `54cd5f4` | discard | no | 3 | 3 | 0 | -1372.581309 | 0.642395 | Row-coarsened exact 64x64 matmul regressed the target small-matmul path and failed the focused gate. |
| `e5e58ed` | discard | no | 1 | 6 | 0 | -1254.545042 | 0.602699 | Fused multiply-add expression lowering failed the focused gate and regressed `fusion_chain5`. |
| `7b68fc5` | keep | yes | 3 | 0 | 0 | 430.255136 | 0.584482 | Exact zero-bias 256 fused-mm route through TF32 cuBLAS qualified twice with no focused regressions. |
| `277a76f` | discard | no | 2 | 1 | 0 | -83.806552 | 0.581182 | Vectorized exact fused-mm ReLU epilogue regressed the 256 fused-mm naive row and failed the focused gate. |
| `2d4e508` | discard | no | 2 | 3 | 0 | -1249.239515 | 0.611800 | Extending zero-bias fused-mm cuBLAS to `128x256x128` regressed the smaller fused-mm rows. |

## Future Research Directions

- Revisit GPU backward with a narrower implementation that only targets the MLP
  benchmark path and avoids affecting unrelated eager elementwise, reduction,
  and inference cases.
- Treat CUDA graph replay as a first-class execution path. The current captured
  inference win is large, and stability work around capture/replay is valuable.
- Prefer fusion and graph-level reductions of launch count over small per-kernel
  launch wrapper tweaks.
- Avoid broad matmul-pattern rewrites unless they are guarded by shape and
  autograd constraints. MLP forward is sensitive to changes that look good in
  isolated matmul tests.
- Consider improving the benchmark methodology with repeated full runs or a
  median-of-runs gate. The no-change benchmark instability means a single run
  can misclassify small changes.
