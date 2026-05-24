## 🧮 **NovaX** — GPU-Accelerated Math for Python

> 🚀 *Explosive performance. Minimal code. One unified tensor library.*

NovaX is a lightweight, modular math library that brings GPU acceleration to standard Python numeric workflows.
It’s designed for speed, simplicity, and extensibility — combining familiar NumPy-style operations with CUDA-backed computation, autograd, and neural-network-ready math ops.

---

### ⚡️ **Features**

* 🔹 **GPU acceleration** via **PyCUDA** and **CUDA-Python**
* 🔹 Clean **NumPy-like API** (`import novax as nx`)
* 🔹 Seamless **CPU ↔ GPU transfer** (`.to_gpu()` / `.to_host()`)
* 🔹 **Elementwise ops**: `add`, `sub`, `mul`, `div`, `pow`, `exp`, `log`, `sqrt`, `abs`, `neg`
* 🔹 **Activation functions**: `relu`, `sigmoid`, `tanh`, `softmax`
* 🔹 **Reduction ops**: `sum`, `mean`, `max`, `min`
* 🔹 **Matrix multiplication** (`matmul` / `@`) with tiled CUDA kernel (16×16 shared memory)
* 🔹 **Automatic differentiation** — reverse-mode autograd with `backward()` and `no_grad()`
* 🔹 Lazy evaluation and **expression graph** with multi-op kernel fusion
* 🔹 **Bucketed GPU memory pool** — O(1) alloc/free with power-of-2 size bins
* 🔹 Adaptive CUDA block size tuned per device
* 🔹 Optional dependency system: install lightweight or GPU-enabled

---

### 🧩 **Installation**

#### CPU-only:

```bash
pip install novax
```

#### GPU-enabled (PyCUDA + CUDA):

```bash
pip install "novax[gpu]"
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/SS-012/novax.git
```

---

### 💻 **Quick Start**

```python
import novax as nx
import numpy as np

# Create tensors
a = nx.Tensor(np.arange(5, dtype=np.float32))
b = nx.Tensor(np.ones(5, dtype=np.float32))

# Elementwise and unary ops
c = nx.relu(a - b)
print(c.data)   # [0. 0. 1. 2. 3.]
```

**Neural network forward + backward pass:**

```python
import novax as nx
import numpy as np

np.random.seed(42)
x  = nx.Tensor(np.random.randn(32, 16).astype(np.float32))
W1 = nx.Tensor(np.random.randn(16, 8).astype(np.float32), requires_grad=True)
b1 = nx.Tensor(np.zeros(8, dtype=np.float32), requires_grad=True)

hidden = nx.relu(nx.matmul(x, W1) + b1)   # (32, 8)
loss   = nx.mean(hidden)
loss.eval().backward()

print(W1.grad.shape)   # (16, 8)
print(b1.grad.shape)   # (8,)
```

**GPU round-trip:**

```python
a = nx.Tensor(np.random.randn(1024).astype(np.float32))
a.to_gpu()
result = nx.exp(a)   # executed as CUDA kernel
print(result.to_host()[:4])
```

---

### ⚙️ **Architecture Overview**

```
novax/
├── core.py          # Tensor class, computation graph, autograd backward()
├── autograd.py      # no_grad context, shared grad-closure factories
├── dispatch.py      # CPU/GPU operation routing with lazy-graph support
├── ops/
│   ├── cpu/         # NumPy fallback implementations (19 ops)
│   ├── gpu/         # PyCUDA kernel wrappers (19 ops)
│   └── launcher.py  # Kernel compiler, fused kernels, matmul, reductions
├── utils/
│   └── mempool.py   # Bucketed GPU memory pool (O(1) alloc/free)
└── __init__.py      # Package entrypoint
```

---

### 🧠 **Why NovaX?**

Traditional Python math libraries (like NumPy) are CPU-bound by default.
NovaX uses GPU memory and fused CUDA kernels to drastically accelerate elementwise operations — while keeping the API as simple as possible.

---

### 🧰 **Roadmap**

* [x] Advanced ops (`matmul`, `exp`, `log`, `pow`, `sqrt`, `abs`, `neg`)
* [x] Activation functions (`relu`, `sigmoid`, `tanh`, `softmax`)
* [x] Reduction ops (`sum`, `mean`, `max`, `min`)
* [x] Fused kernels and expression graph optimization
* [x] Automatic differentiation (reverse-mode autograd)
* [x] Bucketed GPU memory pool (O(1) alloc/free)
* [x] Tiled CUDA matmul kernel (16×16 shared memory)
* [ ] Multi-GPU support
* [ ] Integration with PyTorch tensors
* [ ] Higher-order gradients

---

### 📦 **Build & Test**

```bash
# Build wheel
python -m build

# Run tests
pytest -q
```

---

### 🧑‍💻 **Contributing**

Contributions welcome!

* Open issues or feature requests on [GitHub Issues](https://github.com/SS-012/novax/issues)
* Fork, create a feature branch (`feature/your-feature`), and submit a PR.

---

### 📄 **License**

MIT License © 2025 [Shohaib Shah](https://github.com/SS-012)
You’re free to use, modify, and distribute NovaX under the terms of the MIT License.

---

### 🌟 **Acknowledgments**

* [PyCUDA](https://documen.tician.de/pycuda/) for low-level GPU bindings
* [NumPy](https://numpy.org/) for array semantics inspiration
* [CUDA Python](https://developer.nvidia.com/cuda-python) for kernel execution
