# Math Operations

Elementwise math functions. All functions accept a `Tensor` and return a new
`Tensor`. CPU and GPU execution are handled automatically.

---

## Arithmetic

### `novax.add`

```python
novax.add(a, b) -> Tensor
```

Elementwise addition. Equivalent to `a + b`.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `Tensor` | Left operand |
| `b` | `Tensor` | Right operand |

```python
a = nx.Tensor([1.0, 2.0, 3.0])
b = nx.Tensor([10.0, 20.0, 30.0])
nx.add(a, b).eval().data   # [11. 22. 33.]
```

**Gradient**

$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial \text{out}}, \quad
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial \text{out}}$$

---

### `novax.sub`

```python
novax.sub(a, b) -> Tensor
```

Elementwise subtraction. Equivalent to `a - b`.

```python
nx.sub(a, b).eval().data   # [-9. -18. -27.]
```

**Gradient**

$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial \text{out}}, \quad
\frac{\partial L}{\partial b} = -\frac{\partial L}{\partial \text{out}}$$

---

### `novax.mul`

```python
novax.mul(a, b) -> Tensor
```

Elementwise multiplication. Equivalent to `a * b`.

```python
nx.mul(a, b).eval().data   # [10. 40. 90.]
```

**Gradient**

$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial \text{out}} \cdot b, \quad
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial \text{out}} \cdot a$$

---

### `novax.div`

```python
novax.div(a, b) -> Tensor
```

Elementwise division. Equivalent to `a / b`.

```python
nx.div(a, b).eval().data   # [0.1  0.1  0.1]
```

**Gradient**

$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial \text{out}}  / b, \quad
\frac{\partial L}{\partial b} = -\frac{\partial L}{\partial \text{out}} \cdot \frac{a}{b^2}$$

---

### `novax.pow`

```python
novax.pow(a, b) -> Tensor
```

Elementwise power. Equivalent to `a ** b`.

```python
base = nx.Tensor([2.0, 3.0, 4.0])
exp  = nx.Tensor([3.0, 2.0, 0.5])
nx.pow(base, exp).eval().data   # [8.  9.  2.]
```

**Gradient**

$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial \text{out}} \cdot b \cdot a^{b-1}, \quad
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial \text{out}} \cdot a^b \cdot \ln(a)$$

---

### `novax.neg`

```python
novax.neg(a) -> Tensor
```

Elementwise negation. Equivalent to `-a`.

```python
nx.neg(nx.Tensor([1.0, -2.0, 3.0])).eval().data   # [-1.  2. -3.]
```

**Gradient**

$$\frac{\partial L}{\partial a} = -\frac{\partial L}{\partial \text{out}}$$

---

## Exponential and Logarithm

### `novax.exp`

```python
novax.exp(a) -> Tensor
```

Elementwise natural exponential: $e^a$.

```python
nx.exp(nx.Tensor([0.0, 1.0, 2.0])).eval().data   # [1.     2.718  7.389]
```

GPU kernel: `expf(a[idx])`

**Gradient**

$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial \text{out}} \cdot e^a$$

---

### `novax.log`

```python
novax.log(a) -> Tensor
```

Elementwise natural logarithm: $\ln(a)$.

!!! warning "Domain"
    Input values must be strictly positive. Passing non-positive values
    raises `ValueError` on CPU. GPU behavior is undefined (CUDA returns `-inf` or `nan`).

```python
nx.log(nx.Tensor([1.0, 2.0, 4.0])).eval().data   # [0.     0.693  1.386]
```

GPU kernel: `logf(a[idx])`

**Gradient**

$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial \text{out}} / a$$

---

## Roots and Absolute Value

### `novax.sqrt`

```python
novax.sqrt(a) -> Tensor
```

Elementwise square root: $\sqrt{a}$.

!!! warning "Domain"
    Negative inputs raise `ValueError` on CPU.

```python
nx.sqrt(nx.Tensor([1.0, 4.0, 9.0])).eval().data   # [1. 2. 3.]
```

GPU kernel: `sqrtf(a[idx])`

**Gradient** (numerically stabilised to avoid division by zero)

$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial \text{out}}}{2\sqrt{\max(a,\, 10^{-12})}}$$

---

### `novax.abs`

```python
novax.abs(a) -> Tensor
```

Elementwise absolute value: $|a|$.

```python
nx.abs(nx.Tensor([-3.0, 0.0, 4.0])).eval().data   # [3. 0. 4.]
```

GPU kernel: `fabsf(a[idx])`

**Gradient**

$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial \text{out}} \cdot \text{sign}(a)$$

---

## Broadcasting

All binary ops (`add`, `sub`, `mul`, `div`, `pow`) support broadcasting following
NumPy semantics. When gradients flow backward through a broadcast, NovaX
automatically sums the gradient over the broadcast axes to restore the original shape.

```python
a = nx.Tensor(np.ones((4, 3), dtype=np.float32), requires_grad=True)
b = nx.Tensor(np.array([1.0, 2.0, 3.0]),          requires_grad=True)

loss = nx.sum(a + b)  # b is broadcast from (3,) → (4, 3)
loss.eval().backward()

print(a.grad.shape)   # (4, 3)
print(b.grad.shape)   # (3,)  — summed over broadcast dim
```
