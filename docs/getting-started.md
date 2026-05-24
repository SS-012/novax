# Getting Started

## Requirements

| Requirement | Minimum |
|-------------|---------|
| Python | 3.8+ |
| NumPy | 1.25+ |
| CUDA toolkit *(GPU only)* | 12.0+ |
| PyCUDA *(GPU only)* | 2024.1+ |

---

## Installation

=== "CPU only"

    No GPU dependencies. Uses NumPy for all computations.

    ```bash
    pip install novax
    ```

=== "GPU-enabled"

    Requires a CUDA-capable GPU and the CUDA 12 toolkit already installed on your system.

    ```bash
    pip install "novax[gpu]"
    ```

=== "Development"

    Includes the test suite and coverage tooling.

    ```bash
    git clone https://github.com/SS-012/novax.git
    cd novax
    pip install -e ".[dev]"
    ```

---

## Verify the Install

```python
import novax as nx

print(nx.__version__)        # 0.2.0
print(nx.GPU_AVAILABLE)      # True / False depending on your system
```

---

## Your First Computation

NovaX tensors are created from NumPy arrays, Python lists, or scalars.

```python
import novax as nx
import numpy as np

a = nx.Tensor([1.0, 2.0, 3.0])
b = nx.Tensor([4.0, 5.0, 6.0])

# Arithmetic builds a lazy graph
c = a + b           # no computation yet
print(c.is_leaf)    # False — it's a graph node

# .eval() compiles and executes the graph
result = c.eval()
print(result.data)  # [5. 7. 9.]
```

---

## Selecting the Compute Device

NovaX automatically uses the GPU when tensors are on GPU memory.
Use `to_gpu()` to upload and `to_host()` to download.

```python
if nx.GPU_AVAILABLE:
    a = nx.Tensor(np.ones(1024, dtype=np.float32))
    a.to_gpu()               # now in VRAM

    result = nx.exp(a).eval()
    print(result.to_host())  # back to NumPy array
```

You can override the default device for all operations:

```python
nx.set_default_device("cpu")   # force CPU even when GPU is present
nx.set_default_device("gpu")   # default when GPU is available
```

---

## Gradients in 60 Seconds

Mark tensors with `requires_grad=True`, run a forward pass, call `.backward()`.

```python
import novax as nx
import numpy as np

x = nx.Tensor(np.array([2.0, 3.0]), requires_grad=True)

# y = sum(x^2)  →  dy/dx = 2x
y = nx.sum(x ** 2)
y.eval().backward()

print(x.grad.data)  # [4. 6.]
```

To temporarily disable gradient tracking:

```python
with nx.no_grad():
    z = nx.relu(x).eval()   # no backward closure attached
```

---

## Next Steps

- [**Core Concepts**](concepts.md) — lazy graphs, fusion, memory pool
- [**Tensor API**](api/tensor.md) — full reference for the `Tensor` class
- [**API Reference**](api/math.md) — all ops, activations, reductions, autograd
