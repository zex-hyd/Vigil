"""
Integration tests — deliberately trigger each failure mode.

Requires a CUDA GPU. Run with:
    python tests/integration/test_real_training.py

Each scenario is guarded: the expected exception is caught and the test verifies
that Vigil emitted the correct event_type before the crash.
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
import vigil
from vigil.emitter import Emitter
from vigil.events import TrainingEvent


# ── Shared capture harness ────────────────────────────────────────────────────

class _Capture:
    """Collects TrainingEvents emitted during a test scenario."""

    def __init__(self):
        self.events: list[TrainingEvent] = []

    def __call__(self, batch: list[TrainingEvent]) -> None:
        self.events.extend(batch)

    def wait_for(self, event_type: str, timeout: float = 3.0) -> TrainingEvent | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for e in self.events:
                if e.event_type == event_type:
                    return e
            time.sleep(0.05)
        return None


def _patch_session_emitter(capture: _Capture):
    """Replace the current session's emitter with a capturing one."""
    s = vigil.current_session()
    assert s is not None, "Must be called inside @vigil.watch"
    s._emitter.shutdown(wait=False)
    new_emitter = Emitter(on_events=capture)
    s._emitter = new_emitter
    from vigil.emitter import set_emitter
    set_emitter(new_emitter)
    return new_emitter


# ── Scenario 1: CUDA OOM ──────────────────────────────────────────────────────

def test_oom():
    print("\n[Scenario 1] CUDA OOM")

    capture = _Capture()

    @vigil.watch(project="test-oom")
    def train_oom():
        _patch_session_emitter(capture)
        vigil.watch_model(nn.Linear(1000, 1000).cuda())

        try:
            while True:
                # Allocate growing tensors until CUDA runs out
                x = torch.randn(99_999, 1_000, device="cuda")
                loss = nn.Linear(1000, 1000).cuda()(x).sum()
                loss.backward()
        except torch.cuda.OutOfMemoryError:
            pass  # Expected — Vigil should have captured it via observer hook

        # Give emitter one flush cycle
        time.sleep(0.6)

    train_oom()

    event = capture.wait_for("oom")
    assert event is not None, "Expected an 'oom' event — none captured"
    assert event.project == "test-oom"
    assert event.payload.get("allocated_bytes", 0) > 0, "allocated_bytes should be > 0"
    print(f"  [PASS] OOM event captured at step {event.step}, "
          f"allocated={event.payload['allocated_bytes'] / 1e9:.1f} GiB")


# ── Scenario 2: NaN gradient ──────────────────────────────────────────────────

def test_nan_gradient():
    print("\n[Scenario 2] NaN gradient (lr=1e6)")

    capture = _Capture()

    @vigil.watch(project="test-nan")
    def train_nan():
        model = nn.Linear(10, 1).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e6)
        _patch_session_emitter(capture)
        vigil.watch_model(model)

        for step in range(200):
            x = torch.randn(32, 10, device="cuda")
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            vigil.step()

            nan_event = capture.wait_for("nan_gradient", timeout=0.0)
            if nan_event:
                break  # Got what we came for

        time.sleep(0.6)

    train_nan()

    event = capture.wait_for("nan_gradient")
    assert event is not None, "Expected a 'nan_gradient' event — none captured"
    assert event.project == "test-nan"
    assert "param_name" in event.payload
    print(f"  [PASS] nan_gradient event at step {event.step}, "
          f"param={event.payload['param_name']}")


# ── Scenario 3: Gradient explosion ───────────────────────────────────────────

def test_gradient_explosion():
    print("\n[Scenario 3] Gradient explosion (lr=100, no clip)")

    capture = _Capture()

    @vigil.watch(project="test-explosion")
    def train_explosion():
        model = nn.Linear(512, 512).cuda()
        # Large lr without clipping forces norm >> 100 immediately
        optimizer = torch.optim.SGD(model.parameters(), lr=100.0)
        _patch_session_emitter(capture)
        vigil.watch_model(model, norm_threshold=100.0)  # type: ignore[call-arg]

        for step in range(50):
            x = torch.randn(256, 512, device="cuda")
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            vigil.step()

            expl_event = capture.wait_for("gradient_explosion", timeout=0.0)
            if expl_event:
                break

        time.sleep(0.6)

    train_explosion()

    event = capture.wait_for("gradient_explosion")
    assert event is not None, "Expected a 'gradient_explosion' event — none captured"
    assert event.payload["grad_norm"] > 100.0
    print(f"  [PASS] gradient_explosion at step {event.step}, "
          f"norm={event.payload['grad_norm']:.1f}")


# ── Scenario 4: Dataloader bottleneck ────────────────────────────────────────

def test_dataloader_bottleneck():
    print("\n[Scenario 4] Dataloader bottleneck (sleep=0.2s, num_workers=0)")

    capture = _Capture()

    @vigil.watch(project="test-dataloader")
    def train_slow_dataloader():
        model = nn.Linear(10, 1).cuda()
        _patch_session_emitter(capture)

        class SlowDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 20

            def __getitem__(self, i):
                time.sleep(0.2)  # Deliberate CPU stall
                return torch.randn(10), torch.randn(1)

        raw_loader = torch.utils.data.DataLoader(
            SlowDataset(),
            batch_size=4,
            num_workers=0,  # Single-threaded — no prefetch
        )
        loader = vigil.watch_dataloader(raw_loader)

        for x, y in loader:
            x = x.cuda()
            loss = model(x).sum()
            loss.backward()
            vigil.step()

            btl_event = capture.wait_for("dataloader_bottleneck", timeout=0.0)
            if btl_event:
                break

        time.sleep(0.6)

    train_slow_dataloader()

    event = capture.wait_for("dataloader_bottleneck")
    assert event is not None, (
        "Expected a 'dataloader_bottleneck' event — none captured.\n"
        "Note: pynvml must be installed and GPU utilization must be < 40% during the stall."
    )
    assert event.payload["load_time_s"] > 0.1
    print(f"  [PASS] dataloader_bottleneck at step {event.step}, "
          f"load_time={event.payload['load_time_s']:.3f}s, "
          f"gpu_util={event.payload['gpu_utilization']:.1%}")


# ── Runner ────────────────────────────────────────────────────────────────────

def _require_cuda():
    if not torch.cuda.is_available():
        print("SKIP: No CUDA device available. Run on a GPU machine or Colab.")
        sys.exit(0)


if __name__ == "__main__":
    _require_cuda()

    scenarios = [
        test_oom,
        test_nan_gradient,
        test_gradient_explosion,
        test_dataloader_bottleneck,
    ]

    failed = []
    for fn in scenarios:
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"  [FAIL] {fn.__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
            failed.append(fn.__name__)

    print()
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print(f"All {len(scenarios)} integration scenarios passed.")
