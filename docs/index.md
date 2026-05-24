# NovaX

**GPU-accelerated math library for Python — tensors, neural-network ops, and autograd in one package.**

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **GPU-Accelerated by Default**

    ---

    Every operation dispatches to an optimized CUDA kernel when a GPU is present,
    with transparent CPU fallback via NumPy — no code changes required.

-   :material-brain:{ .lg .middle } **Neural-Network Ready**

    ---

    Activations, reductions, matrix multiplication, and reverse-mode automatic
    differentiation are all first-class citizens.

-   :material-graph:{ .lg .middle } **Lazy Expression Graphs**

    ---

    Operators build a computation graph. Calling `.eval()` compiles and fuses
    the graph into a single CUDA kernel, reducing memory round-trips.

-   :material-memory:{ .lg .middle } **Bucketed Memory Pool**

    ---

    GPU buffers are recycled in O(1) using power-of-2 size bins — allocation
    overhead disappears in tight training loops.

</div>

---

## Install

=== "CPU only"

    ```bash
    pip install novax
    ```

=== "GPU (PyCUDA + CUDA)"

    ```bash
    pip install "novax[gpu]"
    ```

=== "From source"

    ```bash
    git clone https://github.com/SS-012/novax.git
    cd novax
    pip install -e ".[gpu]"
    ```

---

## Quick Examples

### Elementwise math

```python
import novax as nx
import numpy as np

x = nx.Tensor(np.array([1.0, 2.0, 3.0, 4.0]))

# Operator overloading builds a lazy graph; .eval() executes it
y = (x * 2 - 1).eval()
print(y.data)   # [1. 3. 5. 7.]

# Unary ops
print(nx.relu(x - 3).eval().data)    # [0. 0. 0. 1.]
print(nx.exp(nx.log(x)).eval().data) # [1. 2. 3. 4.]
```

### Neural-network forward + backward pass

```python
import novax as nx
import numpy as np

np.random.seed(0)
x  = nx.Tensor(np.random.randn(32, 16).astype(np.float32))
W1 = nx.Tensor(np.random.randn(16, 8).astype(np.float32), requires_grad=True)
b1 = nx.Tensor(np.zeros(8, dtype=np.float32),             requires_grad=True)
W2 = nx.Tensor(np.random.randn(8, 1).astype(np.float32),  requires_grad=True)

# Forward pass
h    = nx.relu(nx.matmul(x, W1) + b1)   # (32, 8)
out  = nx.matmul(h, W2)                  # (32, 1)
loss = nx.mean(out)

# Backward pass
loss.eval().backward()

print(W1.grad.shape)  # (16, 8)
print(W2.grad.shape)  # (8, 1)
print(b1.grad.shape)  # (8,)
```

### GPU round-trip

```python
a = nx.Tensor(np.random.randn(1_000_000).astype(np.float32))
a.to_gpu()                          # upload to VRAM

result = nx.sigmoid(a).eval()       # CUDA kernel, stays on GPU
host   = result.to_host()           # download only when you need it
print(host.shape)                   # (1000000,)
```

---

## Version

Current stable release: **0.2.0** — see the [Changelog](changelog.md) for what's new.
