# Tensor

```python
class novax.Tensor(data, *, requires_grad=False)
```

The core data structure in NovaX. A `Tensor` can hold CPU data (a NumPy array),
GPU data (a PyCUDA device pointer), or be an unevaluated node in a lazy expression
graph.

---

## Constructor

```python
nx.Tensor(data, requires_grad=False)
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `ndarray`, `list`, `float`, `int`, or `None` | Initial data. `None` is used internally for lazy graph nodes. |
| `requires_grad` | `bool` | When `True`, gradients are tracked through operations involving this tensor. Default `False`. |

**Examples**

```python
import novax as nx
import numpy as np

# From a NumPy array
a = nx.Tensor(np.array([1.0, 2.0, 3.0]))

# From a Python list — converted to float32 automatically
b = nx.Tensor([4.0, 5.0, 6.0])

# Scalar (shape becomes (1,))
c = nx.Tensor(3.14)

# With gradient tracking
W = nx.Tensor(np.random.randn(4, 4).astype(np.float32), requires_grad=True)
```

!!! note "dtype"
    All data is stored as `float32`. Integer inputs are silently converted.

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `np.ndarray` or `None` | Host array. `None` when the tensor is on GPU or unevaluated. |
| `shape` | `tuple` | Shape of the tensor data. |
| `size` | `int` | Total number of elements. |
| `dtype` | `np.dtype` | Always `float32`. |
| `on_gpu` | `bool` | `True` when data resides in VRAM. |
| `is_leaf` | `bool` | `True` for tensors created directly from data (not from an op). |
| `requires_grad` | `bool` | Whether gradients are accumulated for this tensor. |
| `grad` | `Tensor` or `None` | Accumulated gradient after calling `backward()`. Initially `None`. |
| `op` | `str` or `None` | The operation that produced this tensor (`"add"`, `"relu"`, …). `None` for leaf tensors. |

---

## Device Management

### `to_gpu()`

```python
tensor.to_gpu() -> Tensor
```

Upload tensor data from host memory to GPU (VRAM). Returns `self` for chaining.

**Raises**

- `RuntimeError` — if no GPU is available (`GPU_AVAILABLE` is `False`)
- `ValueError` — if the tensor has no host data to upload

```python
a = nx.Tensor(np.ones(1024, dtype=np.float32))
a.to_gpu()                 # a.on_gpu is now True, a.data is None
result = nx.exp(a).eval()  # runs CUDA kernel
```

---

### `to_host()`

```python
tensor.to_host() -> np.ndarray
```

Download and return the tensor's data as a NumPy array. Does **not** modify the
tensor — the buffer stays on GPU.

```python
a.to_gpu()
arr = a.to_host()   # np.ndarray, a.on_gpu still True
```

If the tensor is already on CPU, returns `self.data` directly.

---

## Shape Manipulation

### `reshape(*new_shape)`

```python
tensor.reshape(*new_shape) -> Tensor
```

Return a new tensor with data reshaped. Downloads from GPU if necessary.

```python
a = nx.Tensor(np.arange(12, dtype=np.float32))
b = a.reshape(3, 4)
print(b.shape)   # (3, 4)
```

---

### `transpose(axes=None)`

```python
tensor.transpose(axes=None) -> Tensor
```

Return a transposed tensor. Follows NumPy axis convention. Downloads from GPU if necessary.

```python
a = nx.Tensor(np.ones((2, 3), dtype=np.float32))
b = a.transpose()
print(b.shape)   # (3, 2)

# With explicit axis order
c = nx.Tensor(np.ones((2, 3, 4), dtype=np.float32))
d = c.transpose(axes=(2, 0, 1))
print(d.shape)   # (4, 2, 3)
```

---

## Evaluation

### `eval()`

```python
tensor.eval() -> Tensor
```

Compile and execute the expression graph rooted at this tensor. Returns a new
concrete `Tensor` with `data` set (CPU) or `gpu_ptr` set (GPU).

Calling `eval()` on a leaf tensor returns `self` immediately.

```python
a = nx.Tensor([1.0, 2.0])
b = nx.Tensor([3.0, 4.0])

# Graph built lazily
expr = nx.relu(a + b * 2.0)   # no computation yet

# Execute
result = expr.eval()
print(result.data)   # [7. 8.]
```

!!! tip "Always call eval() before backward()"
    `backward()` expects a concrete scalar tensor (the loss). Always call
    `.eval()` on your loss before calling `.backward()`.

---

## Autograd

### `backward()`

```python
tensor.backward() -> None
```

Run reverse-mode automatic differentiation from this tensor.
Seeds `self.grad` with an all-ones tensor, then propagates gradients
backward through the computation graph in topological order.

**Conditions**

- This tensor must be a scalar (or at least the result of a reduction) for
  the default seeding to be meaningful.
- Leaf tensors with `requires_grad=True` will have their `.grad` attribute set.

```python
W = nx.Tensor(np.random.randn(3, 3).astype(np.float32), requires_grad=True)
b = nx.Tensor(np.zeros(3, dtype=np.float32), requires_grad=True)
x = nx.Tensor(np.ones((5, 3), dtype=np.float32))

loss = nx.mean(nx.relu(nx.matmul(x, W) + b))
loss.eval().backward()

print(W.grad.shape)   # (3, 3)
print(b.grad.shape)   # (3,)
```

---

## Operator Overloading

Standard Python arithmetic operators are overloaded to build lazy graph nodes.

| Expression | Operation | Description |
|------------|-----------|-------------|
| `a + b` | add | Elementwise addition |
| `a - b` | sub | Elementwise subtraction |
| `a * b` | mul | Elementwise multiplication |
| `a / b` | div | Elementwise division |
| `a ** b` | pow | Elementwise power |
| `a @ b` | matmul | Matrix multiplication |
| `-a` | neg | Negation |

Scalar operands (`int`, `float`) are automatically wrapped in a `Tensor`.

```python
a = nx.Tensor([1.0, 2.0, 3.0])

(a + 1).eval().data        # [2. 3. 4.]
(a * 0.5).eval().data      # [0.5 1.  1.5]
(2.0 - a).eval().data      # [1.  0. -1.]
(a ** 2).eval().data       # [1. 4. 9.]
(-a).eval().data           # [-1. -2. -3.]
```

---

## Memory Management

### `free(release=False)`

```python
tensor.free(release=False) -> None
```

Return the GPU buffer to the memory pool. Set `release=True` to release memory
back to the CUDA driver rather than caching it.

```python
a.to_gpu()
# ... use a ...
a.free()           # returns buffer to pool for reuse
a.free(release=True)   # returns memory to CUDA driver
```

### Context Manager

`Tensor` can be used as a context manager; the GPU buffer is automatically freed
when the `with` block exits.

```python
with nx.Tensor(np.ones(1024, dtype=np.float32)) as t:
    t.to_gpu()
    result = nx.sigmoid(t).eval()
# t.free() called automatically
```

---

## Representation

```python
>>> nx.Tensor(np.array([1.0, 2.0]), requires_grad=True)
Tensor(shape=(2,), device=CPU, op=None, leaf=True, grad=False)
```
