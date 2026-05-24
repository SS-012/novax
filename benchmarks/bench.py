"""
NovaX vs PyTorch CPU benchmark.
Tests: elementwise ops, activations, reductions, matmul, MLP forward, MLP forward+backward.
"""

import time
import statistics
import numpy as np
import torch
import novax as nx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WARMUP   = 10
RUNS     = 50
SEP      = "-" * 70

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_fn(fn, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1_000  # ms


def ratio(novax_ms, torch_ms):
    r = novax_ms / torch_ms
    indicator = "🐢 SLOWER" if r > 1.05 else ("🚀 FASTER" if r < 0.95 else "≈  TIED")
    return r, indicator


def header(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)
    print(f"{'Operation':<38} {'NovaX':>10} {'PyTorch':>10} {'Ratio':>8}  Result")
    print(SEP)


def row(label, nx_ms, pt_ms):
    r, ind = ratio(nx_ms, pt_ms)
    print(f"{label:<38} {nx_ms:>9.3f}ms {pt_ms:>9.3f}ms {r:>7.2f}x  {ind}")


# ---------------------------------------------------------------------------
# 1. Elementwise ops — vector sizes
# ---------------------------------------------------------------------------
SIZES = [1_000, 100_000, 1_000_000, 10_000_000]

header("ELEMENTWISE OPS")

for sz in SIZES:
    arr = np.random.randn(sz).astype(np.float32)
    a_nx  = nx.Tensor(arr.copy())
    b_nx  = nx.Tensor(arr.copy() + 1.0)
    a_pt  = torch.from_numpy(arr.copy())
    b_pt  = torch.from_numpy(arr.copy() + 1.0)

    def nx_add():  return nx.add(a_nx, b_nx).eval()
    def pt_add():  return a_pt + b_pt
    row(f"add         n={sz:>10,}", time_fn(nx_add), time_fn(pt_add))

    def nx_mul():  return nx.mul(a_nx, b_nx).eval()
    def pt_mul():  return a_pt * b_pt
    row(f"mul         n={sz:>10,}", time_fn(nx_mul), time_fn(pt_mul))

    def nx_exp():  return nx.exp(a_nx).eval()
    def pt_exp():  return torch.exp(a_pt)
    row(f"exp         n={sz:>10,}", time_fn(nx_exp), time_fn(pt_exp))

print(SEP)

# ---------------------------------------------------------------------------
# 2. Activation functions — fixed large size
# ---------------------------------------------------------------------------
header("ACTIVATION FUNCTIONS  (n=1,000,000)")

arr = np.random.randn(1_000_000).astype(np.float32)
a_nx = nx.Tensor(arr.copy())
a_pt = torch.from_numpy(arr.copy())

acts = [
    ("relu",    lambda: nx.relu(a_nx).eval(),    lambda: torch.relu(a_pt)),
    ("sigmoid", lambda: nx.sigmoid(a_nx).eval(), lambda: torch.sigmoid(a_pt)),
    ("tanh",    lambda: nx.tanh(a_nx).eval(),    lambda: torch.tanh(a_pt)),
]
for name, fn_nx, fn_pt in acts:
    row(name, time_fn(fn_nx), time_fn(fn_pt))

arr_soft = np.random.randn(10_000).astype(np.float32)
a_soft_nx = nx.Tensor(arr_soft.copy())
a_soft_pt = torch.from_numpy(arr_soft.copy())
row("softmax  (n=10,000)",
    time_fn(lambda: nx.softmax(a_soft_nx).eval()),
    time_fn(lambda: torch.softmax(a_soft_pt, dim=0)))
print(SEP)

# ---------------------------------------------------------------------------
# 3. Reductions
# ---------------------------------------------------------------------------
header("REDUCTIONS")

for sz in [100_000, 1_000_000, 10_000_000]:
    arr = np.random.randn(sz).astype(np.float32)
    a_nx = nx.Tensor(arr.copy())
    a_pt = torch.from_numpy(arr.copy())

    row(f"sum   n={sz:>10,}", time_fn(lambda: nx.sum(a_nx).eval()),  time_fn(lambda: a_pt.sum()))
    row(f"mean  n={sz:>10,}", time_fn(lambda: nx.mean(a_nx).eval()), time_fn(lambda: a_pt.mean()))
print(SEP)

# ---------------------------------------------------------------------------
# 4. Matrix multiplication
# ---------------------------------------------------------------------------
header("MATRIX MULTIPLICATION")

for M, K, N in [(64, 64, 64), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]:
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    A_nx = nx.Tensor(A.copy())
    B_nx = nx.Tensor(B.copy())
    A_pt = torch.from_numpy(A.copy())
    B_pt = torch.from_numpy(B.copy())

    def nx_mm():  return nx.matmul(A_nx, B_nx).eval()
    def pt_mm():  return A_pt @ B_pt
    row(f"matmul  ({M}x{K}) @ ({K}x{N})", time_fn(nx_mm), time_fn(pt_mm))
print(SEP)

# ---------------------------------------------------------------------------
# 5. MLP forward pass
# ---------------------------------------------------------------------------
header("MLP FORWARD PASS  (batch=128)")

BS = 128
for arch in [(128, 256, 128), (128, 512, 256), (128, 1024, 512)]:
    IN, HID, OUT = arch

    x_arr = np.random.randn(BS, IN).astype(np.float32)
    W1_arr = np.random.randn(IN, HID).astype(np.float32) * 0.02
    b1_arr = np.zeros(HID, dtype=np.float32)
    W2_arr = np.random.randn(HID, OUT).astype(np.float32) * 0.02
    b2_arr = np.zeros(OUT, dtype=np.float32)

    # NovaX
    X_nx  = nx.Tensor(x_arr.copy())
    W1_nx = nx.Tensor(W1_arr.copy())
    b1_nx = nx.Tensor(b1_arr.copy())
    W2_nx = nx.Tensor(W2_arr.copy())
    b2_nx = nx.Tensor(b2_arr.copy())
    def nx_mlp():
        h = nx.relu(nx.matmul(X_nx, W1_nx) + b1_nx)
        o = nx.matmul(h, W2_nx) + b2_nx
        return nx.mean(o).eval()

    # PyTorch
    X_pt  = torch.from_numpy(x_arr.copy())
    W1_pt = torch.from_numpy(W1_arr.copy())
    b1_pt = torch.from_numpy(b1_arr.copy())
    W2_pt = torch.from_numpy(W2_arr.copy())
    b2_pt = torch.from_numpy(b2_arr.copy())
    def pt_mlp():
        h = torch.relu(X_pt @ W1_pt + b1_pt)
        o = h @ W2_pt + b2_pt
        return o.mean()

    row(f"forward  in={IN} hid={HID} out={OUT}", time_fn(nx_mlp), time_fn(pt_mlp))
print(SEP)

# ---------------------------------------------------------------------------
# 6. MLP forward + backward
# ---------------------------------------------------------------------------
header("MLP FORWARD + BACKWARD  (batch=128, 256→128→64)")

IN, HID, OUT = 256, 128, 64

x_arr  = np.random.randn(BS, IN).astype(np.float32)
W1_arr = np.random.randn(IN, HID).astype(np.float32) * 0.02
b1_arr = np.zeros(HID, dtype=np.float32)
W2_arr = np.random.randn(HID, OUT).astype(np.float32) * 0.02
b2_arr = np.zeros(OUT, dtype=np.float32)

# NovaX
X_nx  = nx.Tensor(x_arr.copy())
W1_nx = nx.Tensor(W1_arr.copy(), requires_grad=True)
b1_nx = nx.Tensor(b1_arr.copy(), requires_grad=True)
W2_nx = nx.Tensor(W2_arr.copy(), requires_grad=True)
b2_nx = nx.Tensor(b2_arr.copy(), requires_grad=True)
def nx_fwd_bwd():
    for p in [W1_nx, b1_nx, W2_nx, b2_nx]:
        p.grad = None
    h    = nx.relu(nx.matmul(X_nx, W1_nx) + b1_nx)
    loss = nx.mean(nx.matmul(h, W2_nx) + b2_nx)
    loss.eval().backward()

# PyTorch
X_pt  = torch.from_numpy(x_arr.copy())
W1_pt = torch.tensor(W1_arr.copy(), requires_grad=True)
b1_pt = torch.tensor(b1_arr.copy(), requires_grad=True)
W2_pt = torch.tensor(W2_arr.copy(), requires_grad=True)
b2_pt = torch.tensor(b2_arr.copy(), requires_grad=True)
def pt_fwd_bwd():
    for p in [W1_pt, b1_pt, W2_pt, b2_pt]:
        p.grad = None
    h    = torch.relu(X_pt @ W1_pt + b1_pt)
    loss = (h @ W2_pt + b2_pt).mean()
    loss.backward()

row("forward + backward", time_fn(nx_fwd_bwd), time_fn(pt_fwd_bwd))
print(SEP)

# ---------------------------------------------------------------------------
# 7. Autograd: chain of ops
# ---------------------------------------------------------------------------
header("AUTOGRAD  (chain depth)")

for depth in [5, 10, 20]:
    arr = np.random.randn(10_000).astype(np.float32)

    def nx_chain():
        x = nx.Tensor(arr.copy(), requires_grad=True)
        for _ in range(depth):
            x = nx.relu(x)
        nx.sum(x).eval().backward()

    def pt_chain():
        x = torch.tensor(arr.copy(), requires_grad=True)
        for _ in range(depth):
            x = torch.relu(x)
        x.sum().backward()

    row(f"relu chain  depth={depth}, n=10,000", time_fn(nx_chain, warmup=5, runs=20), time_fn(pt_chain, warmup=5, runs=20))
print(SEP)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"""
LEGEND
  🚀 FASTER  NovaX < 0.95× PyTorch time
  ≈  TIED    within 5%
  🐢 SLOWER  NovaX > 1.05× PyTorch time

ENVIRONMENT
  NovaX  {nx.__version__}  (CPU / NumPy backend — no CUDA in this container)
  PyTorch {torch.__version__}  (CPU)
""")
