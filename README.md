## 🧮 **NovaX** — GPU-Accelerated Math for Python

> 🚀 *Explosive performance. Minimal code. One unified tensor library.*

NovaX is a lightweight, modular math library that brings GPU acceleration to standard Python numeric workflows.
It’s designed for speed, simplicity, and extensibility — combining familiar NumPy-style operations with CUDA-backed computation.

---

### ⚡️ **Features**

* 🔹 **GPU acceleration** via **PyCUDA** and **CUDA-Python**
* 🔹 Clean **NumPy-like API** (`import novax as nx`)
* 🔹 Seamless **CPU ↔ GPU transfer** (`.to_gpu()` / `.to_host()`)
* 🔹 Built-in **elementwise ops**: `add`, `sub`, `mul`, `div`
* 🔹 Lazy evaluation and **expression graph support**
* 🔹 **Extensible kernel system** (add your own CUDA ops)
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
a = nx.Tensor(np.arange(5))
b = nx.Tensor(np.ones(5))

# Upload to GPU
a.to_gpu()
b.to_gpu()

# Perform GPU addition
res = nx.add(a, b)

# Bring results back to host
print(res.to_host())
```

**Output:**

```
[1. 2. 3. 4. 5.]
```

---

### ⚙️ **Architecture Overview**

```
novax/
├── core.py          # Tensor class & computation graph
├── ops/             # CPU and GPU operation modules
│   ├── cpu/         # NumPy fallbacks
│   └── gpu/         # PyCUDA kernels
├── utils/           # Memory management (GPU pool)
├── dispatch.py      # CPU/GPU operation routing
└── __init__.py      # Package entrypoint
```

---

### 🧠 **Why NovaX?**

Traditional Python math libraries (like NumPy) are CPU-bound by default.
NovaX uses GPU memory and fused CUDA kernels to drastically accelerate elementwise operations — while keeping the API as simple as possible.

---

### 🧰 **Roadmap**

* [ ] Advanced ops (matmul, exp, log, pow)
* [ ] Fused kernels and graph optimization
* [ ] Automatic differentiation
* [ ] Multi-GPU support
* [ ] Integration with PyTorch tensors

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
