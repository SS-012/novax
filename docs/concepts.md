# Core Concepts

## Lazy Evaluation and Expression Graphs

NovaX operators do **not** execute immediately. Instead, every arithmetic operation
returns a new `Tensor` that records the operation and its inputs — forming a
directed acyclic graph (DAG).

```python
a = nx.Tensor([1.0, 2.0, 3.0])
b = nx.Tensor([4.0, 5.0, 6.0])

c = a + b       # builds a graph node, no CUDA launched yet
d = c * 2.0     # extends the graph
```

The graph is executed by calling `.eval()`:

```python
result = d.eval()   # compiles graph → executes → returns concrete Tensor
```

This design enables:

1. **Kernel fusion** — a chain of elementwise ops gets compiled into a *single* CUDA kernel
2. **Constant folding** — operations on constants are evaluated at compile time
3. **Device-agnostic code** — the graph records intent; dispatch decides CPU vs GPU at `eval()` time

---

## Kernel Fusion

When `eval()` encounters a chain of elementwise ops, it calls `_build_fused()` to
emit a single CUDA C expression, then compiles and launches **one kernel** instead of many.

```
a + b * exp(c)
```

becomes the single kernel body:

```c
out[idx] = a[idx] + b[idx] * expf(c[idx]);
```

Compiled kernels are cached by `(name, source)` so repeated calls with the same
graph structure pay zero recompilation cost.

---

## Constant Folding

Before building the fused kernel, NovaX walks the graph with `_fold_constants()`.
Subgraphs consisting entirely of scalar constants are collapsed at Python time:

```python
x = nx.Tensor([1.0, 2.0])
y = x + nx.Tensor(2.0) * nx.Tensor(3.0)   # 2.0 * 3.0 folded to 6.0
```

The kernel that reaches the GPU is simply `x[idx] + 6.0f`.

---

## Dispatch: CPU vs GPU

The `dispatch.py` module routes every function call to the correct backend:

```
nx.relu(a)
  └─ _exec_unary("relu", gpu_relu, cpu_relu, a)
       ├─ GPU: if a.on_gpu → launch CUDA kernel
       └─ CPU: np.maximum(0, a.data)
```

Autograd backward closures are attached immediately after execution, before the
result is returned, so the graph is always ready for `backward()`.

---

## Automatic Differentiation

NovaX implements **reverse-mode autodiff** (backpropagation):

1. **Forward pass** — each op attaches a `_backward` closure to the result tensor
   and records the result's dependencies in `_prev`
2. **`.backward()`** — topological sort of `_prev` links, then closures are called
   in reverse order, accumulating gradients via `+=`
3. **Broadcasting** — if a tensor was broadcast during the forward pass, gradients
   are summed back to the original shape

```
loss = mean(relu(matmul(x, W) + b))

Backward graph:
  loss ← mean ← relu ← add ← matmul(x, W)
                         └─── b
```

Only tensors with `requires_grad=True` accumulate gradients. Intermediate nodes
without it are skipped, keeping backward cheap.

---

## GPU Memory Pool

Every GPU buffer allocation goes through `novax.utils.mempool` instead of calling
`cuda.mem_alloc` directly. The pool uses **power-of-2 size bins**:

```
8 B  │ 16 B │ 32 B │ 64 B │ … │ 4 MB │ …
 []       [ptr1]     []      [ptr2, ptr3]
```

When a tensor is freed (`tensor.free()`), its buffer is returned to the matching
bin. The next allocation of the same bucketed size pops the buffer in O(1) without
touching the CUDA driver — eliminating allocation overhead in training loops.

Set `release=True` to return memory to the CUDA driver instead of the pool:

```python
tensor.free(release=True)
```

---

## Thread Safety

The current implementation is **single-threaded**. The global `_grad_enabled` flag
and the module-level kernel cache are not protected by locks. Multi-threaded
training is not supported in v0.2.0.
