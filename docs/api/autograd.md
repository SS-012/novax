# Autograd

NovaX implements **reverse-mode automatic differentiation** (backpropagation).
Gradients are computed lazily: the forward pass builds a computation graph,
and `.backward()` traverses it in reverse to accumulate gradients.

---

## `requires_grad`

Mark a tensor as a **learnable parameter** by setting `requires_grad=True`:

```python
W = nx.Tensor(np.random.randn(8, 4).astype(np.float32), requires_grad=True)
b = nx.Tensor(np.zeros(4, dtype=np.float32),             requires_grad=True)
```

Any operation whose result depends on a `requires_grad=True` tensor will
automatically set `requires_grad=True` on the output and attach a backward closure.

```python
x = nx.Tensor(np.ones((3, 8), dtype=np.float32))  # no grad
h = nx.relu(x @ W + b)                             # h.requires_grad is True
```

---

## `Tensor.grad`

After calling `.backward()`, each leaf tensor with `requires_grad=True` has its
`.grad` attribute set to a `Tensor` containing the accumulated gradient.

```python
W.grad          # Tensor with shape (8, 4)
W.grad.data     # np.ndarray
```

Gradients **accumulate** across multiple backward calls. Zero them manually
between optimisation steps:

```python
# Gradient descent step
for p in [W, b]:
    p.data -= lr * p.grad.data
    p.grad = None   # reset for next iteration
```

---

## `Tensor.backward()`

```python
tensor.backward() -> None
```

Run reverse-mode autodiff from this tensor.

1. Seeds `self.grad = Tensor(np.ones(self.shape))`.
2. Performs a topological sort of the graph via the `_prev` dependency sets.
3. Calls each node's `_backward` closure in reverse topological order, accumulating
   gradients via `+=`.

**Usage**

Call on a scalar (loss) tensor after `.eval()`:

```python
loss = nx.mean(nx.relu(x @ W + b))
loss.eval().backward()
```

!!! warning
    Always call `.eval()` before `.backward()`. `backward()` expects a concrete
    tensor — it cannot be called on a lazy graph node.

---

## `novax.no_grad`

```python
with novax.no_grad():
    ...
```

Context manager that **disables gradient tracking** within its body. Operations
executed inside the block do not attach backward closures, saving memory and
computation.

Useful for inference, validation loops, and when freezing parts of a model.

```python
# Training step — track gradients
h    = nx.relu(nx.matmul(x, W) + b)
loss = nx.mean(h)
loss.eval().backward()

# Inference — no gradients needed
with nx.no_grad():
    pred = nx.softmax(nx.matmul(x_val, W) + b).eval()
```

Gradient tracking is automatically restored when the `with` block exits, even
if an exception is raised.

```python
a = nx.Tensor([1.0], requires_grad=True)
with nx.no_grad():
    c = (a + 1.0).eval()
    print(c.requires_grad)   # False

d = (a + 1.0).eval()
print(d.requires_grad)       # True  — restored
```

---

## Gradient Rules

### Elementwise binary ops

| Op | $\partial L / \partial a$ | $\partial L / \partial b$ |
|----|--------------------------|--------------------------|
| `add` | $g$ | $g$ |
| `sub` | $g$ | $-g$ |
| `mul` | $g \cdot b$ | $g \cdot a$ |
| `div` | $g / b$ | $-g \cdot a / b^2$ |
| `pow` | $g \cdot b \cdot a^{b-1}$ | $g \cdot a^b \cdot \ln(a)$ |

### Unary ops

| Op | $\partial L / \partial a$ |
|----|--------------------------|
| `neg` | $-g$ |
| `exp` | $g \cdot e^a$ |
| `log` | $g / a$ |
| `sqrt` | $g / (2\sqrt{a})$ |
| `abs` | $g \cdot \text{sign}(a)$ |
| `relu` | $g \cdot \mathbf{1}[a > 0]$ |
| `sigmoid` | $g \cdot \sigma(a)(1 - \sigma(a))$ |
| `tanh` | $g \cdot (1 - \tanh^2(a))$ |

### Reductions

| Op | $\partial L / \partial a_i$ |
|----|----------------------------|
| `sum` | $g$ |
| `mean` | $g / n$ |
| `max` | $g \cdot \mathbf{1}[a_i = \max(a)] / \text{count}(\max)$ |
| `min` | $g \cdot \mathbf{1}[a_i = \min(a)] / \text{count}(\min)$ |

### Matmul

$$\frac{\partial L}{\partial A} = g \cdot B^\top, \quad \frac{\partial L}{\partial B} = A^\top \cdot g$$

---

## Broadcasting and Gradients

When a lower-rank tensor is broadcast to a higher-rank shape during a forward op,
NovaX automatically **sums the gradient** back to the original shape during backward.

```python
W = nx.Tensor(np.random.randn(4, 8).astype(np.float32), requires_grad=True)
b = nx.Tensor(np.zeros(8, dtype=np.float32),             requires_grad=True)
x = nx.Tensor(np.ones((32, 4), dtype=np.float32))

# b (shape (8,)) is broadcast across the batch dimension
h = nx.matmul(x, W) + b  # h has shape (32, 8)
nx.mean(h).eval().backward()

print(b.grad.shape)   # (8,)  — NOT (32, 8)
```

The unbroadcast step sums over any leading axes that were added during broadcast
and over any axis where the original size was 1.

---

## Complete Training Loop Example

```python
import novax as nx
import numpy as np

np.random.seed(0)

# Toy dataset: y = sign(sum(x))
X = nx.Tensor(np.random.randn(64, 8).astype(np.float32))
Y = nx.Tensor((np.random.randn(64, 1) > 0).astype(np.float32))

# Parameters
W1 = nx.Tensor(np.random.randn(8, 16).astype(np.float32) * 0.1, requires_grad=True)
b1 = nx.Tensor(np.zeros(16, dtype=np.float32),                   requires_grad=True)
W2 = nx.Tensor(np.random.randn(16, 1).astype(np.float32) * 0.1, requires_grad=True)
b2 = nx.Tensor(np.zeros(1, dtype=np.float32),                    requires_grad=True)

lr = 0.01
params = [W1, b1, W2, b2]

for step in range(100):
    # Forward
    h    = nx.relu(nx.matmul(X, W1) + b1)
    logit = nx.matmul(h, W2) + b2
    pred  = nx.sigmoid(logit)
    loss  = nx.mean((pred - Y) ** 2)  # MSE

    # Backward
    loss.eval().backward()

    # SGD update
    for p in params:
        if p.grad is not None:
            p.data -= lr * p.grad.data
            p.grad = None

    if step % 20 == 0:
        print(f"step {step:3d}  loss={float(loss.data[0]):.4f}")
```
