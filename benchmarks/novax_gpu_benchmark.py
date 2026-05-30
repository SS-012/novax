"""
NovaX vs PyTorch GPU benchmark for autoresearch.

This is a script port of the GPU benchmark notebook. It keeps the same broad
coverage while adding machine-readable output and a focused experiment metric:

    qualified: yes

means the current run improved at least one comparable differentiated-path
NovaX timing versus the baseline JSON and did not regress too many focused
benchmarks. Overall benchmark results are still reported as guardrail context.

Examples:
    python benchmarks/novax_gpu_benchmark.py --profile smoke
    python benchmarks/novax_gpu_benchmark.py --profile research --write-json autoresearch/baseline.json
    python benchmarks/novax_gpu_benchmark.py --profile research --baseline-json autoresearch/baseline.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np


Record = Dict[str, Any]

DIFFERENTIATED_SECTIONS = frozenset({
    "matmul",
    "fusion",
    "fused_mm",
})
DIFFERENTIATED_ID_PREFIXES = (
    "inference_capture_",
)


def is_differentiated_case(record: Record) -> bool:
    """Primary autoresearch target: NovaX paths that can structurally beat PyTorch."""
    case_id = str(record.get("id", ""))
    section = str(record.get("section", ""))
    return section in DIFFERENTIATED_SECTIONS or any(
        case_id.startswith(prefix) for prefix in DIFFERENTIATED_ID_PREFIXES
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark NovaX GPU performance against PyTorch.")
    parser.add_argument(
        "--profile",
        choices=("smoke", "research", "full"),
        default="research",
        help="smoke is fast sanity coverage; research is autoresearch default; full mirrors the notebook sizes.",
    )
    parser.add_argument("--runs", type=int, default=None, help="Measured repetitions per benchmark.")
    parser.add_argument("--warmup", type=int, default=None, help="Warmup repetitions per benchmark.")
    parser.add_argument("--inference-passes", type=int, default=None, help="Passes for repeated inference tests.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for generated inputs.")
    parser.add_argument("--write-json", type=Path, default=None, help="Write all results to this JSON file.")
    parser.add_argument("--baseline-json", type=Path, default=None, help="Compare current NovaX times with this baseline JSON.")
    parser.add_argument("--min-improvement", type=float, default=0.03, help="Minimum timing reduction counted as improved.")
    parser.add_argument("--max-regression", type=float, default=0.05, help="Timing increase counted as a regression.")
    parser.add_argument("--max-regressions", type=int, default=2, help="Absolute regression budget for qualification.")
    parser.add_argument(
        "--max-regression-fraction",
        type=float,
        default=0.10,
        help="Fractional regression budget for qualification.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if any benchmark case errors.")
    parser.add_argument(
        "--fail-on-disqualified",
        action="store_true",
        help="Exit nonzero when baseline comparison does not qualify.",
    )
    return parser.parse_args()


def profile_defaults(profile: str) -> tuple[int, int, int]:
    if profile == "smoke":
        return 3, 7, 50
    if profile == "full":
        return 20, 100, 1000
    return 10, 30, 300


def import_runtime():
    try:
        import torch
    except ImportError:
        sys.exit("PyTorch is not installed. Install it before running this benchmark.")

    if not torch.cuda.is_available():
        sys.exit("PyTorch cannot see a CUDA GPU. Check the CUDA-enabled PyTorch install.")

    try:
        import novax as nx
    except ImportError:
        sys.exit("NovaX is not importable. Run from the repo root or install with: pip install -e .")

    if not getattr(nx, "GPU_AVAILABLE", False):
        sys.exit("NovaX cannot initialize PyCUDA. Install GPU extras with: pip install -e '.[gpu]'")

    try:
        import pycuda.driver as cuda
    except ImportError:
        sys.exit("PyCUDA is not installed. Install NovaX GPU extras first.")

    return torch, nx, cuda


def sync_novax(cuda) -> None:
    cuda.Context.synchronize()


def sync_torch(torch) -> None:
    torch.cuda.synchronize()


def release_novax_output(obj: Any) -> None:
    """Return obvious NovaX GPU outputs to the memory pool after timing."""
    if obj is None:
        return
    if hasattr(obj, "free") and getattr(obj, "on_gpu", False):
        try:
            obj.free()
        except Exception:
            pass
        return
    if isinstance(obj, (tuple, list)):
        for item in obj:
            release_novax_output(item)


def time_novax(fn: Callable[[], Any], cuda, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        out = fn()
        sync_novax(cuda)
        release_novax_output(out)

    times: list[float] = []
    for _ in range(runs):
        sync_novax(cuda)
        t0 = time.perf_counter()
        out = fn()
        sync_novax(cuda)
        times.append(time.perf_counter() - t0)
        release_novax_output(out)
    return statistics.median(times) * 1000.0


def time_torch(fn: Callable[[], Any], torch, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        out = fn()
        sync_torch(torch)
        del out

    times: list[float] = []
    for _ in range(runs):
        sync_torch(torch)
        t0 = time.perf_counter()
        out = fn()
        sync_torch(torch)
        times.append(time.perf_counter() - t0)
        del out
    return statistics.median(times) * 1000.0


def ratio_label(ratio: float) -> str:
    if ratio < 0.90:
        return "FASTER"
    if ratio < 0.95:
        return "FASTER"
    if ratio <= 1.05:
        return "TIED"
    if ratio <= 1.20:
        return "SLOWER"
    return "SLOWER"


def slug(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_")


def print_header(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)
    print(f"{'id':<38} {'NovaX':>10} {'PyTorch':>10} {'Ratio':>8}  Status")
    print("-" * 96)


def run_case(
    records: list[Record],
    *,
    section: str,
    label: str,
    novax_fn: Callable[[], Any],
    torch_fn: Callable[[], Any],
    torch,
    cuda,
    warmup: int,
    runs: int,
    case_id: str | None = None,
) -> Record:
    case_id = case_id or f"{section}_{slug(label)}"
    record: Record = {
        "id": case_id,
        "section": section,
        "label": label,
        "status": "ok",
        "novax_ms": None,
        "torch_ms": None,
        "ratio": None,
        "error": None,
    }

    try:
        novax_ms = time_novax(novax_fn, cuda, warmup, runs)
        torch_ms = time_torch(torch_fn, torch, warmup, runs)
        ratio = novax_ms / torch_ms if torch_ms > 0 else math.inf
        record.update(
            {
                "novax_ms": round(novax_ms, 6),
                "torch_ms": round(torch_ms, 6),
                "ratio": round(ratio, 6),
                "winner": "novax" if ratio < 0.95 else ("tie" if ratio <= 1.05 else "torch"),
            }
        )
        print(f"{case_id:<38} {novax_ms:>9.3f}ms {torch_ms:>9.3f}ms {ratio:>7.3f}x  {ratio_label(ratio)}")
    except Exception as exc:
        record.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
            }
        )
        print(f"{case_id:<38} {'-':>10} {'-':>10} {'-':>8}  ERROR: {type(exc).__name__}: {exc}")

    records.append(record)
    return record


def nx_gpu(nx, arr: np.ndarray):
    t = nx.Tensor(arr.copy())
    t.to_gpu()
    return t


def pt_gpu(torch, arr: np.ndarray, device):
    return torch.from_numpy(arr.copy()).to(device)


def configure_profile(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "elementwise_sizes": [10_000, 100_000],
            "activation_n": 100_000,
            "softmax_n": 10_000,
            "reduction_sizes": [100_000],
            "matmul_shapes": [(64, 64, 64), (256, 256, 256)],
            "mlp_shapes": [(128, 256, 128)],
            "fused_mm_shapes": [(128, 256, 128)],
            "bandwidth_n": 100_000,
        }
    if profile == "full":
        return {
            "elementwise_sizes": [10_000, 100_000, 1_000_000, 10_000_000],
            "activation_n": 1_000_000,
            "softmax_n": 50_000,
            "reduction_sizes": [100_000, 1_000_000, 10_000_000],
            "matmul_shapes": [
                (64, 64, 64),
                (256, 256, 256),
                (512, 512, 512),
                (1024, 1024, 1024),
                (2048, 2048, 2048),
                (4096, 4096, 4096),
            ],
            "mlp_shapes": [(128, 256, 128), (256, 512, 256), (512, 1024, 512)],
            "fused_mm_shapes": [(128, 256, 128), (256, 512, 256), (512, 1024, 512)],
            "bandwidth_n": 10_000_000,
        }
    return {
        "elementwise_sizes": [10_000, 100_000, 1_000_000],
        "activation_n": 1_000_000,
        "softmax_n": 50_000,
        "reduction_sizes": [100_000, 1_000_000],
        "matmul_shapes": [(64, 64, 64), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024)],
        "mlp_shapes": [(128, 256, 128), (256, 512, 256)],
        "fused_mm_shapes": [(128, 256, 128), (256, 512, 256)],
        "bandwidth_n": 1_000_000,
    }


def run_benchmarks(args: argparse.Namespace, torch, nx, cuda, warmup: int, runs: int) -> list[Record]:
    cfg = configure_profile(args.profile)
    device = torch.device("cuda:0")
    records: list[Record] = []

    def case(**kwargs):
        return run_case(records, torch=torch, cuda=cuda, warmup=warmup, runs=runs, **kwargs)

    print_header("ELEMENTWISE OPS")
    for sz in cfg["elementwise_sizes"]:
        arr = np.random.randn(sz).astype(np.float32)
        a_nx = nx_gpu(nx, arr)
        b_nx = nx_gpu(nx, arr + 1.0)
        a_pt = pt_gpu(torch, arr, device)
        b_pt = pt_gpu(torch, arr + 1.0, device)
        case(section="elementwise", label=f"add n={sz}", case_id=f"elementwise_add_n{sz}",
             novax_fn=lambda a=a_nx, b=b_nx: nx.add(a, b).eval(),
             torch_fn=lambda a=a_pt, b=b_pt: a + b)
        case(section="elementwise", label=f"mul n={sz}", case_id=f"elementwise_mul_n{sz}",
             novax_fn=lambda a=a_nx, b=b_nx: nx.mul(a, b).eval(),
             torch_fn=lambda a=a_pt, b=b_pt: a * b)
        case(section="elementwise", label=f"exp n={sz}", case_id=f"elementwise_exp_n{sz}",
             novax_fn=lambda a=a_nx: nx.exp(a).eval(),
             torch_fn=lambda a=a_pt: torch.exp(a))

    print_header("ACTIVATIONS")
    arr = np.random.randn(cfg["activation_n"]).astype(np.float32)
    a_nx = nx_gpu(nx, arr)
    a_pt = pt_gpu(torch, arr, device)
    case(section="activations", label=f"relu n={cfg['activation_n']}", case_id=f"activation_relu_n{cfg['activation_n']}",
         novax_fn=lambda: nx.relu(a_nx).eval(),
         torch_fn=lambda: torch.relu(a_pt))
    case(section="activations", label=f"sigmoid n={cfg['activation_n']}", case_id=f"activation_sigmoid_n{cfg['activation_n']}",
         novax_fn=lambda: nx.sigmoid(a_nx).eval(),
         torch_fn=lambda: torch.sigmoid(a_pt))
    case(section="activations", label=f"tanh n={cfg['activation_n']}", case_id=f"activation_tanh_n{cfg['activation_n']}",
         novax_fn=lambda: nx.tanh(a_nx).eval(),
         torch_fn=lambda: torch.tanh(a_pt))

    arr_s = np.random.randn(cfg["softmax_n"]).astype(np.float32)
    a_s_nx = nx_gpu(nx, arr_s)
    a_s_pt = pt_gpu(torch, arr_s, device)
    case(section="activations", label=f"softmax n={cfg['softmax_n']}", case_id=f"activation_softmax_n{cfg['softmax_n']}",
         novax_fn=lambda: nx.softmax(a_s_nx).eval(),
         torch_fn=lambda: torch.softmax(a_s_pt, dim=0))

    print_header("REDUCTIONS")
    for sz in cfg["reduction_sizes"]:
        arr = np.random.randn(sz).astype(np.float32)
        a_nx = nx_gpu(nx, arr)
        a_pt = pt_gpu(torch, arr, device)
        case(section="reductions", label=f"sum n={sz}", case_id=f"reduction_sum_n{sz}",
             novax_fn=lambda a=a_nx: nx.sum(a).eval(),
             torch_fn=lambda a=a_pt: a.sum())
        case(section="reductions", label=f"mean n={sz}", case_id=f"reduction_mean_n{sz}",
             novax_fn=lambda a=a_nx: nx.mean(a).eval(),
             torch_fn=lambda a=a_pt: a.mean())

    print_header("MATRIX MULTIPLICATION")
    for m, k, n in cfg["matmul_shapes"]:
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)
        a_nx = nx_gpu(nx, a)
        b_nx = nx_gpu(nx, b)
        a_pt = pt_gpu(torch, a, device)
        b_pt = pt_gpu(torch, b, device)
        case(section="matmul", label=f"({m}x{k}) @ ({k}x{n})", case_id=f"matmul_{m}x{k}_x_{k}x{n}",
             novax_fn=lambda a=a_nx, b=b_nx: nx.matmul(a, b).eval(),
             torch_fn=lambda a=a_pt, b=b_pt: a @ b)

    print_header("MLP FORWARD")
    batch_size = 128
    for in_dim, hid_dim, out_dim in cfg["mlp_shapes"]:
        x_a = np.random.randn(batch_size, in_dim).astype(np.float32)
        w1_a = (np.random.randn(in_dim, hid_dim) * 0.02).astype(np.float32)
        b1_a = np.zeros(hid_dim, dtype=np.float32)
        w2_a = (np.random.randn(hid_dim, out_dim) * 0.02).astype(np.float32)
        b2_a = np.zeros(out_dim, dtype=np.float32)

        x_nx = nx_gpu(nx, x_a)
        w1_nx = nx_gpu(nx, w1_a)
        b1_nx = nx_gpu(nx, b1_a)
        w2_nx = nx_gpu(nx, w2_a)
        b2_nx = nx_gpu(nx, b2_a)
        x_pt = pt_gpu(torch, x_a, device)
        w1_pt = pt_gpu(torch, w1_a, device)
        b1_pt = pt_gpu(torch, b1_a, device)
        w2_pt = pt_gpu(torch, w2_a, device)
        b2_pt = pt_gpu(torch, b2_a, device)

        def nx_fwd(x=x_nx, w1=w1_nx, b1=b1_nx, w2=w2_nx, b2=b2_nx):
            h = nx.relu(nx.matmul(x, w1) + b1)
            return nx.mean(nx.matmul(h, w2) + b2).eval()

        def pt_fwd(x=x_pt, w1=w1_pt, b1=b1_pt, w2=w2_pt, b2=b2_pt):
            h = torch.relu(x @ w1 + b1)
            return (h @ w2 + b2).mean()

        case(section="mlp_forward", label=f"in={in_dim} hid={hid_dim} out={out_dim}",
             case_id=f"mlp_forward_{in_dim}_{hid_dim}_{out_dim}",
             novax_fn=nx_fwd, torch_fn=pt_fwd)

    print_header("MLP FORWARD + BACKWARD")
    in_dim, hid_dim, out_dim = 256, 128, 64
    x_a = np.random.randn(batch_size, in_dim).astype(np.float32)
    w1_a = (np.random.randn(in_dim, hid_dim) * 0.02).astype(np.float32)
    b1_a = np.zeros(hid_dim, dtype=np.float32)
    w2_a = (np.random.randn(hid_dim, out_dim) * 0.02).astype(np.float32)
    b2_a = np.zeros(out_dim, dtype=np.float32)

    x_nx = nx_gpu(nx, x_a)
    w1_nx = nx.Tensor(w1_a.copy(), requires_grad=True)
    w1_nx.to_gpu()
    b1_nx = nx.Tensor(b1_a.copy(), requires_grad=True)
    b1_nx.to_gpu()
    w2_nx = nx.Tensor(w2_a.copy(), requires_grad=True)
    w2_nx.to_gpu()
    b2_nx = nx.Tensor(b2_a.copy(), requires_grad=True)
    b2_nx.to_gpu()

    x_pt = pt_gpu(torch, x_a, device)
    w1_pt = torch.tensor(w1_a.copy(), device=device, requires_grad=True)
    b1_pt = torch.tensor(b1_a.copy(), device=device, requires_grad=True)
    w2_pt = torch.tensor(w2_a.copy(), device=device, requires_grad=True)
    b2_pt = torch.tensor(b2_a.copy(), device=device, requires_grad=True)

    def nx_fwd_bwd():
        for p in (w1_nx, b1_nx, w2_nx, b2_nx):
            p.grad = None
        h = nx.relu(nx.matmul(x_nx, w1_nx) + b1_nx)
        loss = nx.mean(nx.matmul(h, w2_nx) + b2_nx)
        loss.eval().backward()

    def pt_fwd_bwd():
        for p in (w1_pt, b1_pt, w2_pt, b2_pt):
            p.grad = None
        h = torch.relu(x_pt @ w1_pt + b1_pt)
        loss = (h @ w2_pt + b2_pt).mean()
        loss.backward()

    case(section="mlp_backward", label="forward + backward", case_id="mlp_forward_backward_256_128_64",
         novax_fn=nx_fwd_bwd, torch_fn=pt_fwd_bwd)

    print_header("KERNEL FUSION")
    arr = np.random.randn(cfg["activation_n"]).astype(np.float32)
    a_nx = nx_gpu(nx, arr)
    b_nx = nx_gpu(nx, arr * 0.5)
    c_nx = nx_gpu(nx, arr + 1.0)
    a_pt = pt_gpu(torch, arr, device)
    b_pt = pt_gpu(torch, arr * 0.5, device)
    c_pt = pt_gpu(torch, arr + 1.0, device)

    case(section="fusion", label="relu(a*b + c)", case_id=f"fusion_chain3_n{cfg['activation_n']}",
         novax_fn=lambda: nx.relu(a_nx * b_nx + c_nx).eval(),
         torch_fn=lambda: torch.relu(a_pt * b_pt + c_pt))
    case(section="fusion", label="sigmoid(relu(a*b+c)*a)", case_id=f"fusion_chain5_n{cfg['activation_n']}",
         novax_fn=lambda: nx.sigmoid(nx.relu(a_nx * b_nx + c_nx) * a_nx).eval(),
         torch_fn=lambda: torch.sigmoid(torch.relu(a_pt * b_pt + c_pt) * a_pt))

    print_header("FUSED MATMUL + BIAS + RELU")
    for m, k, n in cfg["fused_mm_shapes"]:
        x_a = np.random.randn(m, k).astype(np.float32)
        w_a = np.random.randn(k, n).astype(np.float32)
        b_a = np.zeros(n, dtype=np.float32)
        x_nx = nx_gpu(nx, x_a)
        w_nx = nx_gpu(nx, w_a)
        b_nx = nx_gpu(nx, b_a)
        x_pt = pt_gpu(torch, x_a, device)
        w_pt = pt_gpu(torch, w_a, device)
        b_pt = pt_gpu(torch, b_a, device)

        case(section="fused_mm", label=f"({m}x{k}) @ ({k}x{n}) + bias + relu vs naive",
             case_id=f"fused_mm_naive_{m}_{k}_{n}",
             novax_fn=lambda x=x_nx, w=w_nx, b=b_nx: nx.launch_matmul_bias_relu(x, w, b),
             torch_fn=lambda x=x_pt, w=w_pt, b=b_pt: torch.relu(x @ w + b))
        case(section="fused_mm", label=f"({m}x{k}) @ ({k}x{n}) + bias + relu vs linear",
             case_id=f"fused_mm_linear_{m}_{k}_{n}",
             novax_fn=lambda x=x_nx, w=w_nx, b=b_nx: nx.launch_matmul_bias_relu(x, w, b),
             torch_fn=lambda x=x_pt, w=w_pt, b=b_pt: torch.relu(torch.nn.functional.linear(x, w.T, b)))

    print_header("MEMORY BANDWIDTH")
    arr = np.random.randn(cfg["bandwidth_n"]).astype(np.float32)
    a_nx = nx_gpu(nx, arr)
    b_nx = nx_gpu(nx, arr * 0.5)
    a_pt = pt_gpu(torch, arr, device)
    b_pt = pt_gpu(torch, arr * 0.5, device)
    case(section="bandwidth", label=f"add n={cfg['bandwidth_n']}", case_id=f"bandwidth_add_n{cfg['bandwidth_n']}",
         novax_fn=lambda: nx.add(a_nx, b_nx).eval(),
         torch_fn=lambda: a_pt + b_pt)
    case(section="bandwidth", label=f"neg n={cfg['bandwidth_n']}", case_id=f"bandwidth_neg_n{cfg['bandwidth_n']}",
         novax_fn=lambda: nx.neg(a_nx).eval(),
         torch_fn=lambda: -a_pt)
    case(section="bandwidth", label=f"sqrt(abs(a)) n={cfg['bandwidth_n']}", case_id=f"bandwidth_sqrt_abs_n{cfg['bandwidth_n']}",
         novax_fn=lambda: nx.sqrt(nx.abs(a_nx)).eval(),
         torch_fn=lambda: torch.sqrt(torch.abs(a_pt)))

    print_header("REPEATED INFERENCE")
    run_repeated_inference(records, args, torch, nx, cuda, device, runs, warmup)
    return records


def run_repeated_inference(
    records: list[Record],
    args: argparse.Namespace,
    torch,
    nx,
    cuda,
    device,
    runs: int,
    warmup: int,
) -> None:
    passes = args.inference_passes
    in_dim, hid_dim, out_dim, batch_size = 128, 256, 128, 64
    x_a = np.random.randn(batch_size, in_dim).astype(np.float32)
    w1_a = (np.random.randn(in_dim, hid_dim) * 0.02).astype(np.float32)
    b1_a = np.zeros(hid_dim, dtype=np.float32)
    w2_a = (np.random.randn(hid_dim, out_dim) * 0.02).astype(np.float32)
    b2_a = np.zeros(out_dim, dtype=np.float32)

    x_nx = nx_gpu(nx, x_a)
    w1_nx = nx_gpu(nx, w1_a)
    b1_nx = nx_gpu(nx, b1_a)
    w2_nx = nx_gpu(nx, w2_a)
    b2_nx = nx_gpu(nx, b2_a)
    x_pt = pt_gpu(torch, x_a, device)
    w1_pt = pt_gpu(torch, w1_a, device)
    b1_pt = pt_gpu(torch, b1_a, device)
    w2_pt = pt_gpu(torch, w2_a, device)
    b2_pt = pt_gpu(torch, b2_a, device)

    def nx_inf():
        h = nx.relu(nx.matmul(x_nx, w1_nx) + b1_nx)
        return (nx.matmul(h, w2_nx) + b2_nx).eval()

    def pt_inf():
        with torch.no_grad():
            h = torch.relu(x_pt @ w1_pt + b1_pt)
            return h @ w2_pt + b2_pt

    def record_error(case_id: str, label: str, exc: Exception) -> None:
        records.append(
            {
                "id": case_id,
                "section": "inference",
                "label": label,
                "status": "error",
                "novax_ms": None,
                "torch_ms": None,
                "ratio": None,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
            }
        )
        print(f"{case_id:<38} {'-':>10} {'-':>10} {'-':>8}  ERROR: {type(exc).__name__}: {exc}")

    case_id = f"inference_eager_{passes}_passes"
    try:
        for _ in range(warmup):
            release_novax_output(nx_inf())
        sync_novax(cuda)
        t0 = time.perf_counter()
        for _ in range(passes):
            release_novax_output(nx_inf())
        sync_novax(cuda)
        nx_total = (time.perf_counter() - t0) * 1000.0

        for _ in range(warmup):
            _ = pt_inf()
        sync_torch(torch)
        t0 = time.perf_counter()
        for _ in range(passes):
            _ = pt_inf()
        sync_torch(torch)
        pt_total = (time.perf_counter() - t0) * 1000.0

        nx_ms = nx_total / passes
        pt_ms = pt_total / passes
        ratio = nx_ms / pt_ms if pt_ms > 0 else math.inf
        records.append(
            {
                "id": case_id,
                "section": "inference",
                "label": f"eager {passes} passes per-pass avg",
                "status": "ok",
                "novax_ms": round(nx_ms, 6),
                "torch_ms": round(pt_ms, 6),
                "ratio": round(ratio, 6),
                "winner": "novax" if ratio < 0.95 else ("tie" if ratio <= 1.05 else "torch"),
            }
        )
        print(f"{case_id:<38} {nx_ms:>9.3f}ms {pt_ms:>9.3f}ms {ratio:>7.3f}x  {ratio_label(ratio)}")
    except Exception as exc:
        record_error(case_id, f"eager {passes} passes per-pass avg", exc)
        return

    case_id = f"inference_capture_{passes}_passes"
    if not hasattr(nx, "CUDAGraph"):
        records.append(
            {
                "id": case_id,
                "section": "inference",
                "label": f"captured replay {passes} passes per-pass avg",
                "status": "skipped",
                "novax_ms": None,
                "torch_ms": None,
                "ratio": None,
                "error": "NovaX does not expose CUDAGraph",
            }
        )
        print(f"{case_id:<38} {'-':>10} {'-':>10} {'-':>8}  SKIPPED: NovaX does not expose CUDAGraph")
        return

    try:
        graph = nx.CUDAGraph()
        graph.capture(nx_inf)
        for _ in range(warmup):
            graph.replay()
        sync_novax(cuda)
        t0 = time.perf_counter()
        for _ in range(passes):
            graph.replay()
        sync_novax(cuda)
        nx_cap_total = (time.perf_counter() - t0) * 1000.0
        nx_ms = nx_cap_total / passes

        pt_record = next((r for r in records if r["id"] == f"inference_eager_{passes}_passes"), None)
        pt_ms = pt_record["torch_ms"] if pt_record else math.nan
        ratio = nx_ms / pt_ms if pt_ms and pt_ms > 0 else math.inf
        records.append(
            {
                "id": case_id,
                "section": "inference",
                "label": f"captured replay {passes} passes per-pass avg",
                "status": "ok",
                "novax_ms": round(nx_ms, 6),
                "torch_ms": round(pt_ms, 6),
                "ratio": round(ratio, 6),
                "winner": "novax" if ratio < 0.95 else ("tie" if ratio <= 1.05 else "torch"),
            }
        )
        print(f"{case_id:<38} {nx_ms:>9.3f}ms {pt_ms:>9.3f}ms {ratio:>7.3f}x  {ratio_label(ratio)}")
    except Exception as exc:
        record_error(case_id, f"captured replay {passes} passes per-pass avg", exc)


def _pytorch_stats(ok: list[Record]) -> dict[str, Any]:
    ratios = [float(r["ratio"]) for r in ok if r.get("ratio") and r["ratio"] > 0]
    wins = sum(1 for r in ok if r.get("winner") == "novax")
    ties = sum(1 for r in ok if r.get("winner") == "tie")
    losses = sum(1 for r in ok if r.get("winner") == "torch")
    geomean = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else math.nan
    best = min(ok, key=lambda r: r["ratio"]) if ok else None
    return {
        "count": len(ok),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "geomean": round(geomean, 6) if not math.isnan(geomean) else None,
        "best_id": best["id"] if best else None,
        "best_ratio": best["ratio"] if best else None,
    }


def summarize_vs_pytorch(records: list[Record]) -> dict[str, Any]:
    ok = [r for r in records if r.get("status") == "ok" and r.get("ratio")]
    focus_ok = [r for r in ok if is_differentiated_case(r)]
    focus = _pytorch_stats(focus_ok)
    overall = _pytorch_stats(ok)
    return {
        "ok": len(ok),
        "errors": sum(1 for r in records if r.get("status") == "error"),
        "skipped": sum(1 for r in records if r.get("status") == "skipped"),
        "focus_cases": focus["count"],
        "focus_case_ids": [r["id"] for r in focus_ok],
        "pytorch_wins": focus["wins"],
        "pytorch_ties": focus["ties"],
        "pytorch_losses": focus["losses"],
        "geomean_novax_vs_pytorch": focus["geomean"],
        "best_novax_vs_pytorch": focus["best_id"],
        "best_novax_vs_pytorch_ratio": focus["best_ratio"],
        "overall_pytorch_wins": overall["wins"],
        "overall_pytorch_ties": overall["ties"],
        "overall_pytorch_losses": overall["losses"],
        "overall_geomean_novax_vs_pytorch": overall["geomean"],
        "overall_best_novax_vs_pytorch": overall["best_id"],
        "overall_best_novax_vs_pytorch_ratio": overall["best_ratio"],
    }


def load_baseline(path: Path) -> dict[str, Record]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    baseline_records = payload["records"] if isinstance(payload, dict) and "records" in payload else payload
    return {
        r["id"]: r
        for r in baseline_records
        if r.get("status") == "ok" and isinstance(r.get("novax_ms"), (int, float)) and r["novax_ms"] > 0
    }


def summarize_vs_baseline(args: argparse.Namespace, records: list[Record]) -> dict[str, Any] | None:
    if args.baseline_json is None:
        return None
    baseline = load_baseline(args.baseline_json)
    current = {
        r["id"]: r
        for r in records
        if r.get("status") == "ok" and isinstance(r.get("novax_ms"), (int, float)) and r["novax_ms"] > 0
    }
    comparable_ids = sorted(set(baseline) & set(current))
    focused_ids = [case_id for case_id in comparable_ids if is_differentiated_case(current[case_id])]

    def compare(case_ids: list[str]) -> dict[str, Any]:
        improved: list[Record] = []
        regressed: list[Record] = []
        unchanged: list[Record] = []
        positive_log = 0.0
        negative_log = 0.0

        for case_id in case_ids:
            base_ms = float(baseline[case_id]["novax_ms"])
            cur_ms = float(current[case_id]["novax_ms"])
            rel = cur_ms / base_ms
            item = {
                "id": case_id,
                "baseline_ms": round(base_ms, 6),
                "current_ms": round(cur_ms, 6),
                "relative_time": round(rel, 6),
                "speedup": round(base_ms / cur_ms, 6),
            }
            if rel <= 1.0 - args.min_improvement:
                improved.append(item)
                positive_log += math.log(base_ms / cur_ms)
            elif rel >= 1.0 + args.max_regression:
                regressed.append(item)
                negative_log += math.log(cur_ms / base_ms)
            else:
                unchanged.append(item)

        research_score = (
            1000.0 * (positive_log - 1.5 * negative_log)
            + 10.0 * len(improved)
            - 25.0 * len(regressed)
        )
        return {
            "comparable": len(case_ids),
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
            "research_score": research_score,
            "best_improvement": max(improved, key=lambda r: r["speedup"]) if improved else None,
            "worst_regression": max(regressed, key=lambda r: r["relative_time"]) if regressed else None,
        }

    focus = compare(focused_ids)
    overall = compare(comparable_ids)
    regression_budget = max(args.max_regressions, int(math.floor(args.max_regression_fraction * len(focused_ids))))
    improved = focus["improved"]
    regressed = focus["regressed"]
    unchanged = focus["unchanged"]
    research_score = focus["research_score"]
    qualified = len(improved) >= 1 and len(regressed) <= regression_budget and research_score > 0.0

    return {
        "baseline_path": str(args.baseline_json),
        "scope": "differentiated",
        "comparable": focus["comparable"],
        "focus_comparable": focus["comparable"],
        "overall_comparable": overall["comparable"],
        "improved": len(improved),
        "regressed": len(regressed),
        "unchanged": len(unchanged),
        "overall_improved": len(overall["improved"]),
        "overall_regressed": len(overall["regressed"]),
        "overall_unchanged": len(overall["unchanged"]),
        "regression_budget": regression_budget,
        "research_score": round(research_score, 6),
        "qualified": qualified,
        "best_improvement": focus["best_improvement"],
        "worst_regression": focus["worst_regression"],
        "overall_best_improvement": overall["best_improvement"],
        "overall_worst_regression": overall["worst_regression"],
        "improved_cases": improved[:20],
        "regressed_cases": regressed[:20],
        "overall_improved_cases": overall["improved"][:20],
        "overall_regressed_cases": overall["regressed"][:20],
    }


def print_summary(pytorch_summary: dict[str, Any], baseline_summary: dict[str, Any] | None) -> None:
    print()
    print("=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(f"benchmarks_ok: {pytorch_summary['ok']}")
    print(f"benchmarks_error: {pytorch_summary['errors']}")
    print(f"benchmarks_skipped: {pytorch_summary['skipped']}")
    print(f"focus_cases: {pytorch_summary['focus_cases']}")
    print(f"pytorch_wins: {pytorch_summary['pytorch_wins']}")
    print(f"pytorch_ties: {pytorch_summary['pytorch_ties']}")
    print(f"pytorch_losses: {pytorch_summary['pytorch_losses']}")
    print(f"geomean_novax_vs_pytorch: {pytorch_summary['geomean_novax_vs_pytorch']}")
    print(f"best_novax_vs_pytorch: {pytorch_summary['best_novax_vs_pytorch']}")
    print(f"best_novax_vs_pytorch_ratio: {pytorch_summary['best_novax_vs_pytorch_ratio']}")
    print(f"overall_pytorch_wins: {pytorch_summary['overall_pytorch_wins']}")
    print(f"overall_pytorch_ties: {pytorch_summary['overall_pytorch_ties']}")
    print(f"overall_pytorch_losses: {pytorch_summary['overall_pytorch_losses']}")
    print(f"overall_geomean_novax_vs_pytorch: {pytorch_summary['overall_geomean_novax_vs_pytorch']}")

    if baseline_summary is not None:
        print(f"baseline_scope: {baseline_summary['scope']}")
        print(f"baseline_comparable: {baseline_summary['comparable']}")
        print(f"improved_tests: {baseline_summary['improved']}")
        print(f"regressed_tests: {baseline_summary['regressed']}")
        print(f"unchanged_tests: {baseline_summary['unchanged']}")
        print(f"overall_baseline_comparable: {baseline_summary['overall_comparable']}")
        print(f"overall_improved_tests: {baseline_summary['overall_improved']}")
        print(f"overall_regressed_tests: {baseline_summary['overall_regressed']}")
        print(f"overall_unchanged_tests: {baseline_summary['overall_unchanged']}")
        print(f"regression_budget: {baseline_summary['regression_budget']}")
        print(f"research_score: {baseline_summary['research_score']}")
        print(f"qualified: {'yes' if baseline_summary['qualified'] else 'no'}")
        if baseline_summary["best_improvement"]:
            item = baseline_summary["best_improvement"]
            print(f"best_improvement: {item['id']} {item['speedup']}x")
        if baseline_summary["worst_regression"]:
            item = baseline_summary["worst_regression"]
            print(f"worst_regression: {item['id']} {item['relative_time']}x baseline time")
        if baseline_summary["overall_worst_regression"]:
            item = baseline_summary["overall_worst_regression"]
            print(f"overall_worst_regression: {item['id']} {item['relative_time']}x baseline time")
    else:
        print("research_score: n/a")
        print("qualified: n/a (no --baseline-json provided)")


def write_json(
    path: Path,
    args: argparse.Namespace,
    records: list[Record],
    pytorch_summary: dict[str, Any],
    baseline_summary: dict[str, Any] | None,
    torch,
    nx,
    cuda,
    warmup: int,
    runs: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dev = cuda.Device(0)
    payload = {
        "metadata": {
            "profile": args.profile,
            "warmup": warmup,
            "runs": runs,
            "inference_passes": args.inference_passes,
            "seed": args.seed,
            "python": sys.version,
            "platform": platform.platform(),
            "gpu_name": dev.name(),
            "gpu_vram_mb": dev.total_memory() // (1024**2),
            "novax_version": getattr(nx, "__version__", "unknown"),
            "torch_version": torch.__version__,
        },
        "summary": pytorch_summary,
        "baseline_summary": baseline_summary,
        "records": records,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"json_written: {path}")


def main() -> int:
    args = parse_args()
    default_warmup, default_runs, default_passes = profile_defaults(args.profile)
    warmup = args.warmup if args.warmup is not None else default_warmup
    runs = args.runs if args.runs is not None else default_runs
    if args.inference_passes is None:
        args.inference_passes = default_passes

    random.seed(args.seed)
    np.random.seed(args.seed)

    torch, nx, cuda = import_runtime()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dev = cuda.Device(0)
    print("=" * 96)
    print("NovaX GPU benchmark")
    print("=" * 96)
    print(f"profile: {args.profile}")
    print(f"warmup: {warmup}")
    print(f"runs: {runs}")
    print(f"inference_passes: {args.inference_passes}")
    print(f"gpu: {dev.name()}")
    print(f"vram_mb: {dev.total_memory() // (1024**2)}")
    print(f"novax: {getattr(nx, '__version__', 'unknown')}")
    print(f"torch: {torch.__version__}")

    records = run_benchmarks(args, torch, nx, cuda, warmup, runs)
    pytorch_summary = summarize_vs_pytorch(records)
    baseline_summary = summarize_vs_baseline(args, records)
    print_summary(pytorch_summary, baseline_summary)

    if args.write_json is not None:
        write_json(args.write_json, args, records, pytorch_summary, baseline_summary, torch, nx, cuda, warmup, runs)

    if args.strict and pytorch_summary["errors"] > 0:
        return 1
    if args.fail_on_disqualified and baseline_summary is not None and not baseline_summary["qualified"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
