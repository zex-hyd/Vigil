"""
Overhead benchmark — measures Vigil's impact on training step latency.

Target: < 0.5% overhead vs baseline.

Run:
    python tests/integration/test_overhead.py
"""
from __future__ import annotations

import os
import sys
import time

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_root, "sdk"))
sys.path.insert(0, _root)

import torch
import torch.nn as nn

STEPS = 200
WARMUP = 10
BATCH_SIZE = 64
OVERHEAD_LIMIT = 0.5  # percent


def _build_model():
    return nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
    ).cuda()


def benchmark_baseline(steps: int = STEPS) -> list[float]:
    model = _build_model()
    optimizer = torch.optim.Adam(model.parameters())

    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        x = torch.randn(BATCH_SIZE, 512, device="cuda")
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return times


def benchmark_with_vigil(steps: int = STEPS) -> list[float]:
    import vigil
    from vigil.emitter import Emitter

    model = _build_model()
    optimizer = torch.optim.Adam(model.parameters())
    times = []

    # Silence output during benchmark — swap in a no-op emitter
    silent_emitter = Emitter(on_events=lambda events: None)

    @vigil.watch(project="overhead-bench")
    def _run():
        from vigil.emitter import set_emitter
        set_emitter(silent_emitter)

        # Full hook surface: gradient hooks on all params
        vigil.watch_model(model)

        for _ in range(steps):
            t0 = time.perf_counter()
            x = torch.randn(BATCH_SIZE, 512, device="cuda")
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            vigil.step()

    _run()
    return times


def _mean(times: list[float], warmup: int = WARMUP) -> float:
    return sum(times[warmup:]) / len(times[warmup:])


def _p99(times: list[float], warmup: int = WARMUP) -> float:
    import statistics
    trimmed = sorted(times[warmup:])
    idx = int(len(trimmed) * 0.99)
    return trimmed[min(idx, len(trimmed) - 1)]


def run_benchmark():
    if not torch.cuda.is_available():
        print("SKIP: No CUDA device available. Run on a GPU machine or Colab.")
        sys.exit(0)

    device_name = torch.cuda.get_device_name(0)
    print(f"Device:  {device_name}")
    print(f"Steps:   {STEPS}  (first {WARMUP} discarded as warmup)")
    print(f"Batch:   {BATCH_SIZE} × 512")
    print()

    print("Running baseline...")
    baseline_times = benchmark_baseline()

    print("Running with Vigil...")
    vigil_times = benchmark_with_vigil()

    base_mean = _mean(baseline_times)
    vigil_mean = _mean(vigil_times)
    overhead_pct = (vigil_mean - base_mean) / base_mean * 100

    base_p99 = _p99(baseline_times)
    vigil_p99 = _p99(vigil_times)
    p99_overhead_pct = (vigil_p99 - base_p99) / base_p99 * 100

    print()
    print("─" * 44)
    print(f"{'':20s} {'Baseline':>10s}   {'Vigil':>10s}")
    print("─" * 44)
    print(f"{'Mean (ms/step)':20s} {base_mean*1000:>10.3f}   {vigil_mean*1000:>10.3f}")
    print(f"{'p99  (ms/step)':20s} {base_p99*1000:>10.3f}   {vigil_p99*1000:>10.3f}")
    print("─" * 44)
    print(f"{'Mean overhead':20s} {overhead_pct:>10.3f}%")
    print(f"{'p99  overhead':20s} {p99_overhead_pct:>10.3f}%")
    print("─" * 44)
    print()

    status = "PASS" if overhead_pct < OVERHEAD_LIMIT else "FAIL"
    print(f"[{status}] Mean overhead {overhead_pct:.3f}% (limit: {OVERHEAD_LIMIT}%)")

    if overhead_pct >= OVERHEAD_LIMIT:
        sys.exit(1)


if __name__ == "__main__":
    run_benchmark()
