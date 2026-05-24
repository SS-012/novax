# Linear Algebra

---

## `novax.matmul`

```python
novax.matmul(a, b) -> Tensor
```

Matrix multiplication of two 2-D tensors.

$$C = A \cdot B \quad (M \times K) \cdot (K \times N) \to (M \times N)$$

Can also be written using the `@` operator:

```python
C = A @ B
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `Tensor` | Left matrix, shape `(M, K)` |
| `b` | `Tensor` | Right matrix, shape `(K, N)` |

**Returns** `Tensor` of shape `(M, N)`.

**Raises** `ValueError` if either input is not 2-D.

---

### Examples

```python
import novax as nx
import numpy as np

A = nx.Tensor(np.array([[1.0, 2.0],
                         [3.0, 4.0]]))
B = nx.Tensor(np.array([[5.0, 6.0],
                         [7.0, 8.0]]))

C = nx.matmul(A, B)
print(C.eval().data)
# [[19. 22.]
#  [43. 50.]]

# Operator syntax
C2 = (A @ B).eval()
```

---

### GPU kernel — tiled CUDA matmul

On GPU, NovaX uses a **16×16 tiled matrix multiplication** with shared memory:

```c
#define TILE 16
__shared__ float As[TILE][TILE];
__shared__ float Bs[TILE][TILE];

// Each thread block computes a TILE×TILE tile of C.
// Inner loop loads tiles of A and B into shared memory,
// accumulates the dot product, then advances the tile window.
```

Shared memory tiling reduces global memory accesses by a factor of `TILE`
compared to a naive kernel, greatly increasing arithmetic intensity.

Boundary conditions are handled so non-multiple-of-16 shapes work correctly.

---

### Autograd

`matmul` supports full reverse-mode differentiation for both inputs.

**Gradient**

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \cdot B^\top, \quad
\frac{\partial L}{\partial B} = A^\top \cdot \frac{\partial L}{\partial C}$$

```python
A = nx.Tensor(np.random.randn(4, 8).astype(np.float32), requires_grad=True)
B = nx.Tensor(np.random.randn(8, 4).astype(np.float32), requires_grad=True)

loss = nx.mean(A @ B)
loss.eval().backward()

print(A.grad.shape)   # (4, 8)
print(B.grad.shape)   # (8, 4)
```

Only inputs with `requires_grad=True` accumulate a gradient — if `B` is a
frozen weight, set `requires_grad=False` and its gradient will not be computed.

---

## `novax.launch_matmul_bias_relu`

```python
novax.launch_matmul_bias_relu(x, w, bias) -> Tensor
```

**Fused GPU kernel** that performs matmul + bias addition + ReLU activation in a
single CUDA pass, saving two global memory round-trips compared to chaining three
separate ops.

$$\text{out} = \text{relu}(X \cdot W + \text{bias})$$

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `Tensor` | Input, shape `(M, K)`, must be on GPU |
| `w` | `Tensor` | Weights, shape `(K, N)`, must be on GPU |
| `bias` | `Tensor` | Bias vector, shape `(N,)`, must be on GPU |

**Returns** `Tensor` of shape `(M, N)`, on GPU.

!!! warning "GPU only"
    This function requires all three tensors to reside in GPU memory.
    No CPU fallback is provided.

!!! note "Autograd"
    `launch_matmul_bias_relu` is a low-level performance primitive and does not
    attach autograd backward closures. Use it in inference or when you handle
    gradients manually.

```python
if nx.GPU_AVAILABLE:
    M, K, N = 64, 128, 32
    x    = nx.Tensor(np.random.randn(M, K).astype(np.float32)).to_gpu()
    w    = nx.Tensor(np.random.randn(K, N).astype(np.float32)).to_gpu()
    bias = nx.Tensor(np.zeros(N, dtype=np.float32)).to_gpu()

    out = nx.launch_matmul_bias_relu(x, w, bias)
    print(out.to_host().shape)   # (64, 32)
```

---

## Performance Notes

| Matrix size | NovaX GPU (tiled) | NumPy CPU |
|-------------|-------------------|-----------|
| 128 × 128 | ~0.05 ms | ~0.1 ms |
| 512 × 512 | ~0.3 ms | ~4 ms |
| 1024 × 1024 | ~1.2 ms | ~30 ms |

*Indicative timings on an NVIDIA RTX 3080 vs. AMD Ryzen 9 5900X. Actual performance depends heavily on hardware.*

For production-scale matmul, consider cuBLAS (via PyCUDA or PyTorch) which uses
highly tuned GEMM implementations. NovaX's tiled kernel is a clean educational
implementation that already beats NumPy at larger sizes.
