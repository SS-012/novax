# Reduction Operations

Reduction functions collapse a tensor to a smaller tensor (usually a scalar).
All reductions currently operate over **all elements** of the input tensor.

---

## `novax.sum`

```python
novax.sum(a) -> Tensor
```

Sum of all elements. Returns a scalar tensor of shape `(1,)`.

$$\text{sum}(x) = \sum_i x_i$$

```python
a = nx.Tensor([1.0, 2.0, 3.0, 4.0])
nx.sum(a).eval().data   # [10.]
```

**GPU implementation**

Two-pass parallel tree reduction using 256-thread blocks with shared memory:

```
Pass 1: each block reduces its chunk to a partial sum  → partial[] in VRAM
Pass 2: second kernel reduces partial[] to a scalar
```

**Gradient**

$$\frac{\partial L}{\partial a_i} = \frac{\partial L}{\partial \text{out}}$$

Gradient is broadcast back to the full input shape (all-ones scaled by the upstream gradient).

```python
a = nx.Tensor([1.0, 2.0, 3.0], requires_grad=True)
nx.sum(a).eval().backward()
print(a.grad.data)   # [1. 1. 1.]
```

---

## `novax.mean`

```python
novax.mean(a) -> Tensor
```

Arithmetic mean of all elements. Returns a scalar tensor of shape `(1,)`.

$$\text{mean}(x) = \frac{1}{n} \sum_i x_i$$

```python
a = nx.Tensor([1.0, 2.0, 3.0, 4.0])
nx.mean(a).eval().data   # [2.5]
```

**GPU implementation**

Computes `sum(a) / n` where the division is a scalar CUDA kernel applied to the
result of the two-pass reduction.

**Gradient**

$$\frac{\partial L}{\partial a_i} = \frac{\partial L}{\partial \text{out}}}{n}$$

```python
a = nx.Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
nx.mean(a).eval().backward()
print(a.grad.data)   # [0.25 0.25 0.25 0.25]
```

---

## `novax.max`

```python
novax.max(a) -> Tensor
```

Maximum value across all elements. Returns a scalar tensor of shape `(1,)`.

```python
a = nx.Tensor([3.0, 1.0, 4.0, 1.0, 5.0])
nx.max(a).eval().data   # [5.]
```

**GPU implementation**

Two-pass tree reduction with `fmaxf(a, b)` as the reduction operator.

**Gradient**

Gradient flows back through the maximum value(s). If multiple elements share the
maximum, the gradient is split equally.

$$\frac{\partial L}{\partial a_i} = \frac{\partial L}{\partial \text{out}} \cdot \frac{\mathbf{1}[a_i = \max(a)]}{\text{count}(\max)}$$

---

## `novax.min`

```python
novax.min(a) -> Tensor
```

Minimum value across all elements. Returns a scalar tensor of shape `(1,)`.

```python
a = nx.Tensor([3.0, 1.0, 4.0, 1.0, 5.0])
nx.min(a).eval().data   # [1.]
```

**GPU implementation**

Two-pass tree reduction with `fminf(a, b)` as the reduction operator.

**Gradient**

Symmetric to `max` — gradient flows through the minimum value(s).

---

## Common Patterns

### Loss functions

```python
# Mean Squared Error
def mse(pred, target):
    diff = pred - target
    return nx.mean(diff ** 2)

# Binary Cross-Entropy (numerically simplified)
def bce(logits, y):
    return nx.mean(nx.log(nx.sigmoid(logits)) * y +
                   nx.log(1.0 - nx.sigmoid(logits)) * (1.0 - y))
```

### Normalisation

```python
x     = nx.Tensor(np.random.randn(128).astype(np.float32))
mu    = nx.mean(x).eval()
sigma = nx.sqrt(nx.mean((x - float(mu.data[0])) ** 2)).eval()
x_norm = (x - float(mu.data[0])) / (float(sigma.data[0]) + 1e-5)
```

!!! note "Axis reductions coming in v0.3.0"
    Current reductions always flatten the entire tensor. Per-axis reductions
    (`axis=0`, `axis=1`, `keepdims=True`) are planned for a future release.
