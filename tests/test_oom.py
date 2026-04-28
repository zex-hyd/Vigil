"""
test_oom.py — verifies Vigil catches a simulated CUDA OOM event.

Since CI machines may not have a GPU, we deliberately trigger OOM by:
  1. Mocking torch.cuda.OutOfMemoryError via sys.excepthook (as Vigil installs it)
  2. Providing a synthetic OOM capture path that exercises the full pipeline

Run with:
    python tests/test_oom.py
or:
    pytest tests/test_oom.py -v
"""
from __future__ import annotations

import sys
import os
import time
import queue
from dataclasses import asdict

# Make sure the sdk and diagnostic packages are importable
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "sdk"))
# Insert the project root so `diagnostic` is importable as a package
sys.path.insert(0, _root)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_mock_torch():
    """Return a minimal mock of the torch.cuda namespace for non-GPU environments."""
    import types

    torch = types.ModuleType("torch")
    cuda = types.ModuleType("torch.cuda")
    memory = types.ModuleType("torch.cuda.memory")

    cuda.memory_stats = lambda: {
        "allocated_bytes.all.current": 8 * 1024**3,  # 8 GiB
        "reserved_bytes.all.current": 10 * 1024**3,  # 10 GiB
    }
    cuda.memory = memory
    memory._snapshot = lambda: {"segments": []}

    # Minimal OutOfMemoryError
    class OutOfMemoryError(RuntimeError):
        pass

    cuda.OutOfMemoryError = OutOfMemoryError
    torch.cuda = cuda

    # torch._C for observer registration
    _C = types.ModuleType("torch._C")
    torch._C = _C

    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("torch.cuda", cuda)
    sys.modules.setdefault("torch.cuda.memory", memory)
    sys.modules.setdefault("torch._C", _C)

    return torch


# ── Test 1: Low-level OOM capture ────────────────────────────────────────────

def test_oom_event_captured():
    """Vigil emits a TrainingEvent with event_type='oom' when OOM fires."""
    torch = _make_mock_torch()

    captured: list = []

    from vigil.emitter import Emitter
    from vigil.hooks import cuda_hook
    from vigil.events import TrainingEvent

    # Reset installed flag for test isolation
    cuda_hook._installed = False

    step_counter = [0]
    emitter = Emitter(on_events=lambda events: captured.extend(events))

    cuda_hook.install(emitter, project="test-project", step_fn=lambda: step_counter[0])

    # Simulate OOM via sys.excepthook (the path Vigil hooks)
    oom_exc = torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
    sys.excepthook(type(oom_exc), oom_exc, None)

    # Give emitter daemon thread time to flush
    time.sleep(0.6)

    assert len(captured) >= 1, f"Expected at least 1 event, got {len(captured)}"
    event = captured[0]
    assert isinstance(event, TrainingEvent), f"Expected TrainingEvent, got {type(event)}"
    assert event.event_type == "oom", f"Expected event_type='oom', got '{event.event_type}'"
    assert event.project == "test-project"
    assert event.payload["requested_bytes"] == 2 * 1024**3  # 2 GiB

    print(f"[PASS] test_oom_event_captured — event captured: {asdict(event)}")


# ── Test 2: Rule engine fires on OOM event ───────────────────────────────────

def test_rule_engine_oom_diagnosis():
    """Rule oom_grad_accum fires when OOM + batch_size > 64 + grad_accum == 1."""
    from vigil.events import TrainingEvent
    from diagnostic.rule_engine import RuleEngine

    engine = RuleEngine()

    event = TrainingEvent(
        project="test-project",
        step=42,
        event_type="oom",
        payload={
            "allocated_bytes": 8 * 1024**3,
            "reserved_bytes": 10 * 1024**3,
            "requested_bytes": 2 * 1024**3,
            "gradient_accumulation_steps": 1,
            "batch_size": 128,
        },
    )

    result = engine.evaluate(event)
    assert result is not None, "Expected a diagnosis but got None"
    assert result["rule"] == "oom_grad_accum"
    assert result["confidence"] == 0.92
    assert "gradient_accumulation_steps=4" in result["fix"], f"Unexpected fix: {result['fix']}"

    print(f"[PASS] test_rule_engine_oom_diagnosis — diagnosis: {result}")


# ── Test 3: NaN gradient rule ────────────────────────────────────────────────

def test_rule_engine_nan_gradient():
    """Rule nan_loss_no_warmup fires for NaN gradient with no scheduler and high loss spike."""
    from vigil.events import TrainingEvent
    from diagnostic.rule_engine import RuleEngine

    engine = RuleEngine()

    event = TrainingEvent(
        project="test-project",
        step=10,
        event_type="nan_gradient",
        payload={
            "param_name": "transformer.layer.0.weight",
            "grad_norm": float("nan"),
            "lr_scheduler": None,
            "loss_spike_ratio": 15.0,
        },
    )

    result = engine.evaluate(event)
    assert result is not None, "Expected a diagnosis but got None"
    assert result["rule"] == "nan_loss_no_warmup"
    assert result["confidence"] == 0.85

    print(f"[PASS] test_rule_engine_nan_gradient — diagnosis: {result}")


# ── Test 4: Dataloader bottleneck rule ───────────────────────────────────────

def test_rule_engine_dataloader_bottleneck():
    """Rule dataloader_bottleneck fires when num_workers <= 1."""
    from vigil.events import TrainingEvent
    from diagnostic.rule_engine import RuleEngine

    engine = RuleEngine()

    event = TrainingEvent(
        project="test-project",
        step=5,
        event_type="dataloader_bottleneck",
        payload={
            "load_time_s": 0.45,
            "gpu_utilization": 0.12,
            "num_workers": 0,
        },
    )

    result = engine.evaluate(event)
    assert result is not None, "Expected a diagnosis but got None"
    assert result["rule"] == "dataloader_bottleneck"
    assert result["confidence"] == 0.90
    assert "num_workers=" in result["fix"]

    print(f"[PASS] test_rule_engine_dataloader_bottleneck — diagnosis: {result}")


# ── Test 5: No rule fires when conditions not met ────────────────────────────

def test_rule_engine_no_match():
    """No rule fires when OOM occurs but batch_size <= 64."""
    from vigil.events import TrainingEvent
    from diagnostic.rule_engine import RuleEngine

    engine = RuleEngine()

    event = TrainingEvent(
        project="test-project",
        step=1,
        event_type="oom",
        payload={
            "gradient_accumulation_steps": 1,
            "batch_size": 32,  # not > 64, rule should not fire
        },
    )

    result = engine.evaluate(event)
    assert result is None, f"Expected no diagnosis but got: {result}"

    print("[PASS] test_rule_engine_no_match — correctly returned None")


# ── Test 6: Emitter never blocks under back-pressure ────────────────────────

def test_emitter_nonblocking():
    """queue.put_nowait() must never raise even when queue is full."""
    from vigil.emitter import Emitter, _MAX_QUEUE
    from vigil.events import TrainingEvent

    emitter = Emitter(on_events=lambda events: None)

    # Flood the queue well past its limit
    for i in range(_MAX_QUEUE + 200):
        event = TrainingEvent(
            project="test-project",
            step=i,
            event_type="oom",
            payload={},
        )
        emitter.emit(event)  # must not block or raise

    emitter.shutdown(wait=False)
    print("[PASS] test_emitter_nonblocking — no block or exception under flood")


# ── Test 7: End-to-end @vigil.watch integration ──────────────────────────────

def test_watch_decorator_integration():
    """@vigil.watch wires up the session; vigil.step() advances counter."""
    # Ensure torch mock is present
    _make_mock_torch()

    sys.path.insert(0, os.path.join(_root, "sdk"))

    import vigil

    results = []

    @vigil.watch(project="integration-test")
    def fake_train():
        for i in range(3):
            vigil.step()
            # Manually emit a test event through the session's emitter
            from vigil.events import TrainingEvent
            s = vigil.current_session()
            event = TrainingEvent(
                project="integration-test",
                step=s._step,
                event_type="dataloader_bottleneck",
                payload={
                    "load_time_s": 0.3,
                    "gpu_utilization": 0.1,
                    "num_workers": 0,
                },
            )
            s._emitter.emit(event)

        time.sleep(0.6)  # let emitter flush

    # Patch _handle_event to collect results
    original_handle = vigil._TrainingSession._handle_event

    def capturing_handle(self, event):
        results.append(event)
        original_handle(self, event)

    vigil._TrainingSession._handle_event = capturing_handle
    try:
        fake_train()
    finally:
        vigil._TrainingSession._handle_event = original_handle

    assert len(results) == 3, f"Expected 3 events, got {len(results)}"
    assert all(e.event_type == "dataloader_bottleneck" for e in results)
    print(f"[PASS] test_watch_decorator_integration — {len(results)} events captured end-to-end")


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_oom_event_captured,
        test_rule_engine_oom_diagnosis,
        test_rule_engine_nan_gradient,
        test_rule_engine_dataloader_bottleneck,
        test_rule_engine_no_match,
        test_emitter_nonblocking,
        test_watch_decorator_integration,
    ]

    failed = []
    for t in tests:
        try:
            t()
        except Exception as exc:
            print(f"[FAIL] {t.__name__}: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            failed.append(t.__name__)

    print()
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests passed.")
