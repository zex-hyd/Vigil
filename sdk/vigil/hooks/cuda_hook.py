"""CUDA OOM hook: patches the torch memory allocator to capture snapshots on OOM."""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vigil.emitter import Emitter

_ORIG_MALLOC = None
_installed = False

# Skip duplicate OOM emits when allocator observer + handled-exception path both fire.
_LAST_OOM_EMIT_NS = 0
_OOM_EMIT_DEBOUNCE_NS = 150_000_000


def _training_payload_from_session() -> dict:
    """Merge ``@watch(config={...})`` into OOM payloads so rule engine sees batch_size etc."""
    try:
        import vigil as _v

        s = _v.current_session()
        if s is None:
            return {}
        cfg = getattr(s, "_config", None)
        return dict(cfg) if cfg else {}
    except Exception:
        return {}


def install(emitter: "Emitter", project: str, step_fn) -> None:
    """
    Monkey-patch torch.cuda's allocator error path to intercept OOM exceptions.

    Because torch raises OutOfMemoryError (a subclass of RuntimeError) from C++,
    we wrap sys.excepthook and also install an allocator hook via
    torch.cuda.memory._record_memory_history when available.

    The simplest portable approach that never blocks: wrap the __exit__ of the
    training context at the Python level by patching torch.Tensor.backward and
    torch.cuda.synchronize is fragile. Instead, we use torch's built-in
    _cuda_setAllocatorSettings is not available publicly, so we rely on
    exception interception via threading.excepthook + sys.excepthook.
    """
    global _installed
    if _installed:
        return
    _installed = True

    try:
        import torch
    except ImportError:
        return

    _patch_torch_allocator(emitter, project, step_fn, torch)


def _patch_torch_allocator(emitter, project, step_fn, torch) -> None:
    """
    Install a CUDA memory snapshot capture on OutOfMemoryError.

    We hook into torch.cuda by wrapping the caching allocator's error path.
    PyTorch 2.0+ exposes torch.cuda.memory._snapshot() for heap snapshots.
    We install via threading.excepthook so we capture OOM from any thread.
    """
    import threading

    orig_excepthook = threading.excepthook

    def _vigil_thread_excepthook(args):
        if _is_cuda_oom(args.exc_value):
            _capture_oom(emitter, project, step_fn, args.exc_value, torch)
        orig_excepthook(args)

    threading.excepthook = _vigil_thread_excepthook

    orig_sys_excepthook = sys.excepthook

    def _vigil_sys_excepthook(exc_type, exc_value, exc_tb):
        if _is_cuda_oom(exc_value):
            _capture_oom(emitter, project, step_fn, exc_value, torch)
        orig_sys_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _vigil_sys_excepthook

    # Also wrap torch.cuda.memory.empty_cache to get snapshot just before the
    # allocator gives up. PyTorch 2.0+ lets us register an OOM observer.
    _try_register_oom_observer(emitter, project, step_fn, torch)


def _try_register_oom_observer(emitter, project, step_fn, torch) -> None:
    """Use torch._C._cuda_attach_out_of_memory_observer if available (PyTorch 2.1+)."""
    try:
        observer_fn = getattr(torch._C, "_cuda_attach_out_of_memory_observer", None)
        if observer_fn is None:
            return

        def _oom_observer(device, alloc, device_alloc, device_free):
            _capture_oom_raw(
                emitter=emitter,
                project=project,
                step_fn=step_fn,
                torch=torch,
                requested_bytes=alloc,
                allocated_bytes=device_alloc,
                free_bytes=device_free,
            )

        observer_fn(_oom_observer)
    except Exception:
        pass


def capture_if_cuda_oom(emitter: "Emitter", project: str, step_fn, exc: BaseException) -> None:
    """When the user catches ``torch.cuda.OutOfMemoryError`` outside ``@watch``, ``sys.excepthook``
    never runs — call this from ``session.run`` so OOM events still enqueue."""
    try:
        import torch
    except ImportError:
        return
    if not _is_cuda_oom(exc):
        return
    try:
        _capture_oom(emitter, project, step_fn, exc, torch)
        # Deliver on caller thread — Colab/Jupyter often hides daemon-thread prints.
        emitter.flush_now()
    except Exception as e:
        import sys

        print(f"Vigil: OOM capture/flush failed: {e!r}", file=sys.stderr, flush=True)


def _is_cuda_oom(exc) -> bool:
    if exc is None:
        return False
    # torch.cuda.OutOfMemoryError is a subclass of RuntimeError
    exc_type_name = type(exc).__name__
    if exc_type_name == "OutOfMemoryError":
        return True
    msg = str(exc).lower()
    if "out of memory" in msg and ("cuda" in msg or "cublas" in msg or "cudnn" in msg):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in msg:
        return True
    return False


def _capture_oom(emitter, project, step_fn, exc, torch) -> None:
    try:
        mem_stats = torch.cuda.memory_stats()
        allocated = mem_stats.get("allocated_bytes.all.current", 0)
        reserved = mem_stats.get("reserved_bytes.all.current", 0)
        # Parse requested bytes from the error message as a fallback
        requested = _parse_requested_bytes(str(exc))
    except Exception:
        allocated = reserved = requested = 0

    _capture_oom_raw(
        emitter=emitter,
        project=project,
        step_fn=step_fn,
        torch=torch,
        requested_bytes=requested,
        allocated_bytes=allocated,
        free_bytes=reserved,
    )


def _capture_oom_raw(emitter, project, step_fn, torch, requested_bytes, allocated_bytes, free_bytes) -> None:
    global _LAST_OOM_EMIT_NS
    import time

    now = time.time_ns()
    if now - _LAST_OOM_EMIT_NS < _OOM_EMIT_DEBOUNCE_NS:
        return
    _LAST_OOM_EMIT_NS = now

    from vigil.events import TrainingEvent

    snapshot = None
    try:
        if hasattr(torch.cuda.memory, "_snapshot"):
            snapshot_data = torch.cuda.memory._snapshot()
            # snapshot is large; store summary only
            snapshot = {"segments": len(snapshot_data.get("segments", []))}
    except Exception:
        pass

    extras = _training_payload_from_session()
    payload = dict(extras)
    payload.update(
        {
            "allocated_bytes": allocated_bytes,
            "reserved_bytes": free_bytes,
            "requested_bytes": requested_bytes,
            "snapshot_summary": snapshot,
        }
    )

    event = TrainingEvent(
        project=project,
        step=step_fn(),
        event_type="oom",
        payload=payload,
    )
    emitter.emit(event)


def _parse_requested_bytes(msg: str) -> int:
    """Extract 'Tried to allocate X GiB/MiB' from OOM message."""
    import re

    m = re.search(r"Tried to allocate ([\d.]+) ([GMK]iB)", msg)
    if not m:
        return 0
    amount, unit = float(m.group(1)), m.group(2)
    multipliers = {"KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}
    return int(amount * multipliers.get(unit, 1))
