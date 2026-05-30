# NovaX Autoresearch Findings

This document summarizes the NovaX GPU optimization research logged in
`autoresearch/results.tsv` as of May 27, 2026.

## Count Summary

- Logged rows: 62
- Baseline setup rows: 1
- Experiment evaluations after baseline: 61
- Unique non-baseline experiment commits: 60
- Currently successful unique experiments: 14
- Strict benchmark-qualified performance successes: 1
- Currently discarded or reverted unique experiments: 46

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
- NovaX wins vs PyTorch: 10
- NovaX ties vs PyTorch: 3
- PyTorch wins vs NovaX: 22
- Geomean NovaX/PyTorch time: 1.046350
- Best NovaX case vs PyTorch: `inference_capture_300_passes`
- Best ratio: 0.170820

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
