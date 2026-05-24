# Activation Functions

Pointwise nonlinearities commonly used as hidden-layer activations in neural networks.

---

## `novax.relu`

```python
novax.relu(a) -> Tensor
```

Rectified Linear Unit. Sets negative values to zero.

$$\text{relu}(x) = \max(0,\, x)$$

```python
a = nx.Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
nx.relu(a).eval().data   # [0. 0. 0. 1. 2.]
```

GPU kernel: `fmaxf(0.0f, a[idx])`

**Gradient**

$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial \text{out}} \cdot \mathbf{1}[a > 0]$$

The gradient is zero for all non-positive inputs (the "dead neuron" problem does
not get special handling in NovaX — that is left to the caller's initialisation
and learning-rate choices).

---

## `novax.sigmoid`

```python
novax.sigmoid(a) -> Tensor
```

Logistic sigmoid. Squashes values into $(0, 1)$.

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

```python
a = nx.Tensor([-2.0, 0.0, 2.0])
nx.sigmoid(a).eval().data   # [0.119  0.5    0.881]
```

GPU kernel: `1.0f / (1.0f + expf(-a[idx]))`

**Gradient**

$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial \text{out}} \cdot \sigma(a)\,(1 - \sigma(a))$$

---

## `novax.tanh`

```python
novax.tanh(a) -> Tensor
```

Hyperbolic tangent. Squashes values into $(-1, 1)$.

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

```python
a = nx.Tensor([-1.0, 0.0, 1.0])
nx.tanh(a).eval().data   # [-0.762  0.     0.762]
```

GPU kernel: `tanhf(a[idx])`

**Gradient**

$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial \text{out}} \cdot (1 - \tanh^2(a))$$

---

## `novax.softmax`

```python
novax.softmax(a) -> Tensor
```

Softmax normalisation over all elements of the tensor. Produces a probability
distribution that sums to 1.

$$\text{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

The $-\max(x)$ shift is applied for **numerical stability** — it prevents
overflow in the exponential without changing the output.

```python
logits = nx.Tensor([2.0, 1.0, 0.1])
probs  = nx.softmax(logits).eval()
print(probs.data)              # [0.659  0.242  0.099]
print(probs.data.sum())        # 1.0
```

!!! note "Axis"
    The current implementation always reduces over **all** elements.
    Per-row softmax (axis=1) is not yet supported.

**GPU implementation**

On GPU, softmax is computed in three passes:

1. Parallel reduction to find `max(x)`
2. Elementwise `exp(x - max)` kernel
3. Parallel reduction to find the normalisation sum, followed by a division kernel

**Gradient**

Softmax backward is not yet implemented in the autograd engine. Call it in
inference-only contexts, or outside a `requires_grad` scope.

```python
with nx.no_grad():
    probs = nx.softmax(logits).eval()
```

---

## Choosing an Activation

| Activation | Output range | Common use |
|------------|-------------|------------|
| `relu` | $[0, +\infty)$ | Hidden layers, CNNs |
| `sigmoid` | $(0, 1)$ | Binary classification output |
| `tanh` | $(-1, 1)$ | RNNs, normalised hidden layers |
| `softmax` | $(0, 1)$, sum = 1 | Multiclass classification output |

---

## Example — Two-layer MLP

```python
import novax as nx
import numpy as np

np.random.seed(42)
x  = nx.Tensor(np.random.randn(32, 16).astype(np.float32))
W1 = nx.Tensor(np.random.randn(16, 8).astype(np.float32), requires_grad=True)
b1 = nx.Tensor(np.zeros(8, dtype=np.float32),             requires_grad=True)
W2 = nx.Tensor(np.random.randn(8, 4).astype(np.float32),  requires_grad=True)
b2 = nx.Tensor(np.zeros(4, dtype=np.float32),             requires_grad=True)

# Hidden layer with ReLU, output with sigmoid
h   = nx.relu(nx.matmul(x, W1) + b1)    # (32, 8)
out = nx.sigmoid(nx.matmul(h, W2) + b2)  # (32, 4)

loss = nx.mean(out)
loss.eval().backward()
```
