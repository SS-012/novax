"""
NovaX vs PyTorch GPU benchmark.

Requires: CUDA GPU, PyCUDA, PyTorch with CUDA support.

Run:
    pip install "novax[gpu]" torch
    python benchmarks/bench_gpu.py

Tests (all on GPU, float32 unless noted):
  1.  Elementwise ops — various sizes
  2.  Activation functions
  3.  Reductions
  4.  Matrix multiplication — 64 to 4096
  5.  MLP forward pass
  6.  MLP forward + backward
  7.  Kernel fusion — fused vs unfused chain
  8.  Fused matmul+bias+relu kernel
  9.  Memory throughput (bandwidth-bound ops)
  10. Warm model inference (fixed-shape repeated forward)
"""

import sys
import time
import statistics
import numpy as np

# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

try:
    import torch
    if not torch.cuda.is_available():
        sys.exit("PyTorch cannot see a CUDA GPU. Check your CUDA installation.")
except ImportError:
    sys.exit("PyTorch not installed. Run: pip install torch")

try:
    import novax as nx
    if not nx.GPU_AVAILABLE:
        sys.exit("NovaX cannot initialise PyCUDA. "
                 "Install the GPU extras: pip install 'novax[gpu]'")
except ImportError:
    sys.exit("NovaX not installed. Run: pip install -e .")

import pycuda.driver as cuda

# ---------------------------------------------------------------------------
# Device info
# ---------------------------------------------------------------------------

dev = cuda.Device(0)
print("\n" + "=" * 70)
print("  GPU BENCHMARK: NovaX vs PyTorch")
print("=" * 70)
print(f"  GPU            : {dev.name()}")
print(f"  VRAM           : {dev.total_memory() // (1024**2):,} MB")
print(f"  NovaX          : {nx.__version__}")
print(f"  PyTorch        : {torch.__version__}")
print("=" * 70 + "\n")

torch_device = torch.device("cuda:0")

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

WARMUP = 20
RUNS   = 100
SEP    = "-" * 70

def time_novax(fn, warmup=WARMUP, runs=RUNS):
    """Time a NovaX GPU function; sync before each measurement."""
    for _ in range(warmup):
        fn()
    cuda.Context.synchronize()
    times = []
    for _ in range(runs):
        cuda.Context.synchronize()
        t0 = time.perf_counter()
        fn()
        cuda.Context.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1_000  # ms


def time_torch(fn, warmup=WARMUP, runs=RUNS):
    """Time a PyTorch CUDA function; sync before each measurement."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1_000


def ratio(novax_ms, torch_ms):
    r = novax_ms / torch_ms
    if r < 0.90:
        ind = "🚀 FASTER"
    elif r < 0.95:
        ind = "🟢 FASTER "
    elif r <= 1.05:
        ind = "≈  TIED  "
    elif r <= 1.20:
        ind = "🟡 SLOWER"
    else:
        ind = "🔴 SLOWER"
    return r, ind


def header(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)
    print(f"{'Operation':<40} {'NovaX':>9} {'PyTorch':>9} {'Ratio':>7}  Result")
    print(SEP)


def row(label, nx_ms, pt_ms):
    r, ind = ratio(nx_ms, pt_ms)
    print(f"{label:<40} {nx_ms:>8.3f}ms {pt_ms:>8.3f}ms {r:>6.2f}x  {ind}")


def nx_gpu(arr):
    """Upload a numpy array to a NovaX GPU tensor."""
    t = nx.Tensor(arr.copy())
    t.to_gpu()
    return t


def pt_gpu(arr):
    """Upload a numpy array to a PyTorch CUDA tensor."""
    return torch.from_numpy(arr.copy()).to(torch_device)


# ---------------------------------------------------------------------------
# 1. Elementwise ops
# ---------------------------------------------------------------------------
header("ELEMENTWISE OPS")

for sz in [10_000, 100_000, 1_000_000, 10_000_000]:
    arr = np.random.randn(sz).astype(np.float32)
    a_nx = nx_gpu(arr)
    b_nx = nx_gpu(arr + 1.0)
    a_pt = pt_gpu(arr)
    b_pt = pt_gpu(arr + 1.0)

    row(f"add          n={sz:>10,}",
        time_novax(lambda: nx.add(a_nx, b_nx).eval()),
        time_torch(lambda: a_pt + b_pt))

    row(f"mul          n={sz:>10,}",
        time_novax(lambda: nx.mul(a_nx, b_nx).eval()),
        time_torch(lambda: a_pt * b_pt))

    row(f"exp          n={sz:>10,}",
        time_novax(lambda: nx.exp(a_nx).eval()),
        time_torch(lambda: torch.exp(a_pt)))

print(SEP)

# ---------------------------------------------------------------------------
# 2. Activations
# ---------------------------------------------------------------------------
header("ACTIVATION FUNCTIONS  (n=1,000,000)")

arr = np.random.randn(1_000_000).astype(np.float32)
a_nx = nx_gpu(arr)
a_pt = pt_gpu(arr)

for name, fn_nx, fn_pt in [
    ("relu",    lambda: nx.relu(a_nx).eval(),    lambda: torch.relu(a_pt)),
    ("sigmoid", lambda: nx.sigmoid(a_nx).eval(), lambda: torch.sigmoid(a_pt)),
    ("tanh",    lambda: nx.tanh(a_nx).eval(),    lambda: torch.tanh(a_pt)),
]:
    row(name, time_novax(fn_nx), time_torch(fn_pt))

# Softmax on smaller vector (per-row semantics differ; use 1D for fair comparison)
arr_s = np.random.randn(50_000).astype(np.float32)
a_s_nx = nx_gpu(arr_s)
a_s_pt = pt_gpu(arr_s)
row("softmax  (n=50,000)",
    time_novax(lambda: nx.softmax(a_s_nx).eval()),
    time_torch(lambda: torch.softmax(a_s_pt, dim=0)))
print(SEP)

# ---------------------------------------------------------------------------
# 3. Reductions
# ---------------------------------------------------------------------------
header("REDUCTIONS")

for sz in [100_000, 1_000_000, 10_000_000]:
    arr = np.random.randn(sz).astype(np.float32)
    a_nx = nx_gpu(arr)
    a_pt = pt_gpu(arr)

    row(f"sum    n={sz:>10,}",
        time_novax(lambda: nx.sum(a_nx).eval()),
        time_torch(lambda: a_pt.sum()))
    row(f"mean   n={sz:>10,}",
        time_novax(lambda: nx.mean(a_nx).eval()),
        time_torch(lambda: a_pt.mean()))
print(SEP)

# ---------------------------------------------------------------------------
# 4. Matrix multiplication
# ---------------------------------------------------------------------------
header("MATRIX MULTIPLICATION")

for M, K, N in [(64, 64, 64), (256, 256, 256), (512, 512, 512),
                (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]:
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    A_nx = nx_gpu(A); B_nx = nx_gpu(B)
    A_pt = pt_gpu(A); B_pt = pt_gpu(B)

    row(f"matmul  ({M}×{K}) @ ({K}×{N})",
        time_novax(lambda: nx.matmul(A_nx, B_nx).eval()),
        time_torch(lambda: A_pt @ B_pt))
print(SEP)

# ---------------------------------------------------------------------------
# 5. MLP forward pass
# ---------------------------------------------------------------------------
header("MLP FORWARD PASS  (batch=128)")

BS = 128
for IN, HID, OUT in [(128, 256, 128), (256, 512, 256), (512, 1024, 512)]:
    x_a  = np.random.randn(BS, IN).astype(np.float32)
    W1_a = np.random.randn(IN, HID).astype(np.float32) * 0.02
    b1_a = np.zeros(HID, dtype=np.float32)
    W2_a = np.random.randn(HID, OUT).astype(np.float32) * 0.02
    b2_a = np.zeros(OUT, dtype=np.float32)

    X_nx  = nx_gpu(x_a);  W1_nx = nx_gpu(W1_a); b1_nx = nx_gpu(b1_a)
    W2_nx = nx_gpu(W2_a); b2_nx = nx_gpu(b2_a)
    X_pt  = pt_gpu(x_a);  W1_pt = pt_gpu(W1_a); b1_pt = pt_gpu(b1_a)
    W2_pt = pt_gpu(W2_a); b2_pt = pt_gpu(b2_a)

    def nx_fwd():
        h = nx.relu(nx.matmul(X_nx, W1_nx) + b1_nx)
        return nx.mean(nx.matmul(h, W2_nx) + b2_nx).eval()

    def pt_fwd():
        h = torch.relu(X_pt @ W1_pt + b1_pt)
        return (h @ W2_pt + b2_pt).mean()

    row(f"fwd  in={IN} hid={HID} out={OUT}", time_novax(nx_fwd), time_torch(pt_fwd))
print(SEP)

# ---------------------------------------------------------------------------
# 6. MLP forward + backward
# ---------------------------------------------------------------------------
header("MLP FORWARD + BACKWARD  (batch=128, 256→128→64)")

IN, HID, OUT = 256, 128, 64
x_a  = np.random.randn(BS, IN).astype(np.float32)
W1_a = np.random.randn(IN, HID).astype(np.float32) * 0.02
b1_a = np.zeros(HID, dtype=np.float32)
W2_a = np.random.randn(HID, OUT).astype(np.float32) * 0.02
b2_a = np.zeros(OUT, dtype=np.float32)

X_nx  = nx_gpu(x_a)
W1_nx = nx.Tensor(W1_a.copy(), requires_grad=True); W1_nx.to_gpu()
b1_nx = nx.Tensor(b1_a.copy(), requires_grad=True); b1_nx.to_gpu()
W2_nx = nx.Tensor(W2_a.copy(), requires_grad=True); W2_nx.to_gpu()
b2_nx = nx.Tensor(b2_a.copy(), requires_grad=True); b2_nx.to_gpu()

X_pt  = pt_gpu(x_a)
W1_pt = torch.tensor(W1_a.copy(), device=torch_device, requires_grad=True)
b1_pt = torch.tensor(b1_a.copy(), device=torch_device, requires_grad=True)
W2_pt = torch.tensor(W2_a.copy(), device=torch_device, requires_grad=True)
b2_pt = torch.tensor(b2_a.copy(), device=torch_device, requires_grad=True)

def nx_fwd_bwd():
    for p in [W1_nx, b1_nx, W2_nx, b2_nx]:
        p.grad = None
    h    = nx.relu(nx.matmul(X_nx, W1_nx) + b1_nx)
    loss = nx.mean(nx.matmul(h, W2_nx) + b2_nx)
    loss.eval().backward()

def pt_fwd_bwd():
    for p in [W1_pt, b1_pt, W2_pt, b2_pt]:
        p.grad = None
    h    = torch.relu(X_pt @ W1_pt + b1_pt)
    loss = (h @ W2_pt + b2_pt).mean()
    loss.backward()

row("forward + backward", time_novax(nx_fwd_bwd), time_torch(pt_fwd_bwd))
print(SEP)

# ---------------------------------------------------------------------------
# 7. Kernel fusion: fused vs unfused chain (NovaX internal)
# ---------------------------------------------------------------------------
header("KERNEL FUSION  (elementwise chain, n=1,000,000)")

arr = np.random.randn(1_000_000).astype(np.float32)
a_nx = nx_gpu(arr); b_nx = nx_gpu(arr * 0.5); c_nx = nx_gpu(arr + 1.0)
a_pt = pt_gpu(arr); b_pt = pt_gpu(arr * 0.5); c_pt = pt_gpu(arr + 1.0)

# 3-op chain: relu(a * b + c)  →  NovaX fuses to 1 kernel, PyTorch launches 3
def nx_chain3():
    return nx.relu(nx.mul(a_nx, b_nx) + c_nx).eval()
def pt_chain3():
    return torch.relu(a_pt * b_pt + c_pt)

row("relu(a*b + c)  [3 ops → 1 kernel]", time_novax(nx_chain3), time_torch(pt_chain3))

# 5-op chain
def nx_chain5():
    t = nx.relu(nx.mul(a_nx, b_nx) + c_nx)
    return nx.sigmoid(t * a_nx).eval()
def pt_chain5():
    t = torch.relu(a_pt * b_pt + c_pt)
    return torch.sigmoid(t * a_pt)

row("sigmoid(relu(a*b+c)*a)  [5 ops]", time_novax(nx_chain5), time_torch(pt_chain5))
print(SEP)

# ---------------------------------------------------------------------------
# 8. Fused matmul+bias+relu kernel
# ---------------------------------------------------------------------------
header("FUSED MATMUL+BIAS+RELU")

for M, K, N in [(128, 256, 128), (256, 512, 256), (512, 1024, 512)]:
    X_a  = np.random.randn(M, K).astype(np.float32)
    W_a  = np.random.randn(K, N).astype(np.float32)
    b_a  = np.zeros(N, dtype=np.float32)

    X_nx = nx_gpu(X_a); W_nx = nx_gpu(W_a); b_nx_f = nx_gpu(b_a)
    X_pt = pt_gpu(X_a); W_pt = pt_gpu(W_a); b_pt_f = pt_gpu(b_a)

    # NovaX: single fused kernel
    def nx_fused():
        return nx.launch_matmul_bias_relu(X_nx, W_nx, b_nx_f)

    # PyTorch: three separate ops (typical usage)
    def pt_unfused():
        return torch.relu(X_pt @ W_pt + b_pt_f)

    # PyTorch: torch.nn.functional.linear (also fused internally via cuBLAS + pointwise)
    def pt_linear():
        return torch.relu(torch.nn.functional.linear(X_pt, W_pt.T, b_pt_f))

    row(f"({M}×{K})@({K}×{N})+bias+relu  NovaX fused vs pt naive",
        time_novax(nx_fused), time_torch(pt_unfused))
    row(f"({M}×{K})@({K}×{N})+bias+relu  NovaX fused vs pt linear",
        time_novax(nx_fused), time_torch(pt_linear))
print(SEP)

# ---------------------------------------------------------------------------
# 9. Memory bandwidth: large copy / fill
# ---------------------------------------------------------------------------
header("MEMORY BANDWIDTH  (copy-heavy ops, n=10,000,000)")

arr = np.random.randn(10_000_000).astype(np.float32)
a_nx = nx_gpu(arr); b_nx = nx_gpu(arr * 0.5)
a_pt = pt_gpu(arr); b_pt = pt_gpu(arr * 0.5)

for name, fn_nx, fn_pt in [
    ("add  (reads 2, writes 1)",
        lambda: nx.add(a_nx, b_nx).eval(),
        lambda: a_pt + b_pt),
    ("neg  (reads 1, writes 1)",
        lambda: nx.neg(a_nx).eval(),
        lambda: -a_pt),
    ("sqrt (reads 1, writes 1)",
        lambda: nx.sqrt(nx.abs(a_nx)).eval(),
        lambda: torch.sqrt(torch.abs(a_pt))),
]:
    row(name, time_novax(fn_nx), time_torch(fn_pt))

mb = arr.nbytes / 1024**2
nx_bw_t = time_novax(lambda: nx.add(a_nx, b_nx).eval()) / 1000
pt_bw_t = time_torch(lambda: a_pt + b_pt) / 1000
print(f"\n  Effective bandwidth (add, 3 arrays × {mb:.0f} MB):")
print(f"    NovaX  : {3*mb/nx_bw_t/1024:.1f} GB/s")
print(f"    PyTorch: {3*mb/pt_bw_t/1024:.1f} GB/s")
print(SEP)

# ---------------------------------------------------------------------------
# 10. Inference throughput: repeated fixed-shape forward pass
# ---------------------------------------------------------------------------
header("REPEATED INFERENCE (1000 forward passes, batch=64, 128→256→128)")

IN, HID, OUT = 128, 256, 128
x_a  = np.random.randn(64, IN).astype(np.float32)
W1_a = np.random.randn(IN, HID).astype(np.float32) * 0.02
b1_a = np.zeros(HID, dtype=np.float32)
W2_a = np.random.randn(HID, OUT).astype(np.float32) * 0.02
b2_a = np.zeros(OUT, dtype=np.float32)

X_nx  = nx_gpu(x_a);  W1_nx = nx_gpu(W1_a); b1_nx = nx_gpu(b1_a)
W2_nx = nx_gpu(W2_a); b2_nx = nx_gpu(b2_a)
X_pt  = pt_gpu(x_a);  W1_pt = pt_gpu(W1_a); b1_pt = pt_gpu(b1_a)
W2_pt = pt_gpu(W2_a); b2_pt = pt_gpu(b2_a)

def nx_inf():
    h = nx.relu(nx.matmul(X_nx, W1_nx) + b1_nx)
    return (nx.matmul(h, W2_nx) + b2_nx).eval()

def pt_inf():
    with torch.no_grad():
        h = torch.relu(X_pt @ W1_pt + b1_pt)
        return h @ W2_pt + b2_pt

N_INF = 1000

# --- (a) eager loop: rebuild + dispatch the graph every iteration ----------
cuda.Context.synchronize()
t0 = time.perf_counter()
for _ in range(N_INF):
    nx_inf()
cuda.Context.synchronize()
nx_total = (time.perf_counter() - t0) * 1000

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_INF):
    pt_inf()
torch.cuda.synchronize()
pt_total = (time.perf_counter() - t0) * 1000

r, ind = ratio(nx_total / N_INF, pt_total / N_INF)
print(f"{'eager: 1000× fwd (per-pass avg)':<40} {nx_total/N_INF:>8.3f}ms {pt_total/N_INF:>8.3f}ms {r:>6.2f}x  {ind}")
print(f"  Throughput: NovaX {N_INF/(nx_total/1000):.0f} fwd/s  |  PyTorch {N_INF/(pt_total/1000):.0f} fwd/s")

# --- (b) captured replay: build once, replay the whole graph ---------------
# This is NovaX's structural advantage — the entire forward pass becomes a
# single graph launch with zero per-op Python. Compared against PyTorch EAGER
# (the typical inference loop without torch.compile / CUDA graphs).
graph = nx.CUDAGraph()
graph.capture(nx_inf)          # warm + capture (no-op fallback if unsupported)
for _ in range(WARMUP):
    graph.replay()
cuda.Context.synchronize()
t0 = time.perf_counter()
for _ in range(N_INF):
    graph.replay()
cuda.Context.synchronize()
nx_cap_total = (time.perf_counter() - t0) * 1000

r2, ind2 = ratio(nx_cap_total / N_INF, pt_total / N_INF)
print(f"{'captured: 1000× fwd replay (avg)':<40} {nx_cap_total/N_INF:>8.3f}ms {pt_total/N_INF:>8.3f}ms {r2:>6.2f}x  {ind2}")
print(f"  Throughput: NovaX {N_INF/(nx_cap_total/1000):.0f} fwd/s (captured)  |  PyTorch {N_INF/(pt_total/1000):.0f} fwd/s (eager)")
print(f"  Speedup from capture/replay vs NovaX eager: {nx_total/nx_cap_total:.2f}×")
print(SEP)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"""
LEGEND
  🚀 FASTER   NovaX < 0.90× PyTorch time
  🟢 FASTER   NovaX 0.90–0.95× PyTorch time
  ≈  TIED     within 5%
  🟡 SLOWER   NovaX 1.05–1.20× PyTorch time
  🔴 SLOWER   NovaX > 1.20× PyTorch time

Note on kernel fusion (section 7):
  NovaX compiles a chain of elementwise ops into ONE CUDA kernel.
  PyTorch (without torch.compile) launches a separate kernel per op.
  Watch for 🚀 FASTER results in that section — that is NovaX's
  core structural advantage over stock PyTorch.
""")
