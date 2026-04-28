"""
Vigil — ML training observability SDK.

Usage:
    import vigil

    @vigil.watch(project="my-project")
    def train():
        model = MyModel()
        for epoch in range(100):
            loss = train_step(batch)
            loss.backward()
            optimizer.step()
"""
from __future__ import annotations

import functools
import sys
from typing import Callable, TypeVar

from vigil.emitter import Emitter, set_emitter
from vigil.events import TrainingEvent

# Add the project root (parent of sdk/) to sys.path so `diagnostic` is importable
# whether the user runs from within sdk/ or from the monorepo root.
import sys as _sys
import pathlib as _pathlib
_project_root = str(_pathlib.Path(__file__).resolve().parents[2])
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

try:
    from diagnostic.rule_engine import RuleEngine
    _rule_engine_available = True
    _rule_engine_import_error: str | None = None
except ImportError as e:
    _rule_engine_available = False
    _rule_engine_import_error = f"diagnostic import failed ({e!r}) — add repo root to sys.path, e.g. sys.path.insert(0, 'Vigil')"

F = TypeVar("F", bound=Callable)

__all__ = ["watch", "TrainingEvent"]
__version__ = "0.1.0"


def watch(
    project: str,
    norm_threshold: float = 100.0,
    auto_wrap_dataloaders: bool = True,
):
    """
    Decorator that instruments a training function with Vigil observability hooks.

    Args:
        project: Project identifier shown in diagnostics output.
        norm_threshold: Gradient norm above which an explosion event is fired.
        auto_wrap_dataloaders: If True, wraps DataLoader objects found in local
            scope after the first forward pass. Set False for manual wrapping.
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            session = _TrainingSession(
                project=project,
                norm_threshold=norm_threshold,
                auto_wrap_dataloaders=auto_wrap_dataloaders,
            )
            return session.run(fn, args, kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator


class _TrainingSession:
    """Manages the lifecycle of a single watched training run."""

    def __init__(self, project: str, norm_threshold: float, auto_wrap_dataloaders: bool):
        self.project = project
        self.norm_threshold = norm_threshold
        self.auto_wrap_dataloaders = auto_wrap_dataloaders
        self._step = 0
        self._rule_engine = RuleEngine() if _rule_engine_available else None
        if self._rule_engine is None and _rule_engine_import_error:
            print(f"Vigil: {_rule_engine_import_error}", file=sys.stderr, flush=True)

        def _on_events(events: list[TrainingEvent]) -> None:
            for event in events:
                self._handle_event(event)

        self._emitter = Emitter(on_events=_on_events)
        set_emitter(self._emitter)

        # Install CUDA OOM hook immediately (before training starts)
        self._install_cuda_hook()

    def _step_fn(self) -> int:
        return self._step

    def _install_cuda_hook(self) -> None:
        from vigil.hooks import cuda_hook
        cuda_hook.install(self._emitter, self.project, self._step_fn)

    def install_gradient_hooks(self, model, norm_threshold: float | None = None) -> None:
        from vigil.hooks import gradient_hook
        nt = self.norm_threshold if norm_threshold is None else norm_threshold
        gradient_hook.install(model, self._emitter, self.project, self._step_fn, nt)

    def wrap_dataloader(self, dataloader):
        from vigil.hooks import dataloader_profiler
        return dataloader_profiler.wrap(dataloader, self._emitter, self.project, self._step_fn)

    def _handle_event(self, event: TrainingEvent) -> None:
        if self._rule_engine is not None:
            diagnosis = self._rule_engine.evaluate(event)
            if diagnosis:
                _print_diagnosis(event, diagnosis)
                return
        # No matching rule — emit raw JSON for debugging
        import json
        from dataclasses import asdict
        print(json.dumps(asdict(event)), flush=True)

    def run(self, fn: Callable, args: tuple, kwargs: dict):
        # Inject session into function's global scope so hooks can self-register
        # The recommended pattern: user calls vigil.current_session() inside train()
        _push_session(self)
        try:
            return fn(*args, **kwargs)
        finally:
            _pop_session()
            self._emitter.shutdown(wait=True)


# ── Session stack (allows nested watch decorators, though uncommon) ─────────

_session_stack: list[_TrainingSession] = []


def _push_session(s: _TrainingSession) -> None:
    _session_stack.append(s)


def _pop_session() -> None:
    if _session_stack:
        _session_stack.pop()


def current_session() -> _TrainingSession | None:
    return _session_stack[-1] if _session_stack else None


# ── Convenience helpers users call inside their training function ────────────

def watch_model(model, norm_threshold: float | None = None) -> None:
    """Register gradient hooks on all model parameters. Call after model init.

    Pass ``norm_threshold=float('inf')`` to disable gradient explosion / NaN hooks
    (e.g. pure OOM stress tests).
    """
    s = current_session()
    if s is None:
        raise RuntimeError("vigil.watch_model() called outside of a @vigil.watch decorated function")
    s.install_gradient_hooks(model, norm_threshold=norm_threshold)


def watch_dataloader(dataloader):
    """Wrap a DataLoader with the Vigil profiler. Returns the wrapped loader."""
    s = current_session()
    if s is None:
        raise RuntimeError("vigil.watch_dataloader() called outside of a @vigil.watch decorated function")
    return s.wrap_dataloader(dataloader)


def step(n: int = 1) -> None:
    """Advance the step counter. Call once per optimizer step."""
    s = current_session()
    if s is not None:
        s._step += n


# ── Diagnosis printer ────────────────────────────────────────────────────────

def _print_diagnosis(event: TrainingEvent, diagnosis: dict) -> None:
    border = "━" * 40
    print(border)
    print(f"Vigil  |  {event.project}  |  step {event.step}")
    print(border)
    print(f"Error:   {event.event_type}")
    print(f"Cause:   {diagnosis['diagnosis']}")
    print(f"Fix:     {diagnosis['fix']}")
    print(f"Confidence: {diagnosis['confidence']:.0%}")
    print(border)
    sys.stdout.flush()
