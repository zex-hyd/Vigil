"""Dataloader profiler: measures batch load time and GPU utilization via pynvml."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from vigil.emitter import Emitter

_GPU_UTIL_THRESHOLD = 0.4   # 40% utilisation
_LOAD_TIME_THRESHOLD = 0.1  # 100 ms
# Emit at most one dataloader_bottleneck event per this many seconds (global per wrapped loader).
_BOTTLENECK_COOLDOWN_S = 30.0


def wrap(dataloader, emitter: "Emitter", project: str, step_fn) -> "_ProfiledDataLoader":
    return _ProfiledDataLoader(dataloader, emitter, project, step_fn)


class _ProfiledDataLoader:
    """Transparent wrapper around any DataLoader-like iterable."""

    def __init__(self, dataloader, emitter: "Emitter", project: str, step_fn):
        self._dl = dataloader
        self._emitter = emitter
        self._project = project
        self._step_fn = step_fn
        self._nvml = _NvmlHandle()
        self._last_bottleneck_mono: float = 0.0

    def __len__(self):
        return len(self._dl)

    def __iter__(self) -> Iterator:
        return _ProfiledIterator(
            iter(self._dl),
            self._emitter,
            self._project,
            self._step_fn,
            self._nvml,
            self,
        )

    def __getattr__(self, name):
        return getattr(self._dl, name)


class _ProfiledIterator:
    def __init__(
        self,
        inner_iter,
        emitter,
        project,
        step_fn,
        nvml: "_NvmlHandle",
        parent_loader: "_ProfiledDataLoader",
    ):
        self._iter = inner_iter
        self._emitter = emitter
        self._project = project
        self._step_fn = step_fn
        self._nvml = nvml
        self._parent = parent_loader
        self._t_start: float | None = None

    def __iter__(self):
        return self

    def __next__(self):
        t0 = time.perf_counter()
        batch = next(self._iter)  # raises StopIteration naturally
        load_time = time.perf_counter() - t0

        try:
            self._check_bottleneck(load_time)
        except Exception:
            pass

        return batch

    def _check_bottleneck(self, load_time: float) -> None:
        if load_time <= _LOAD_TIME_THRESHOLD:
            return

        gpu_util = self._nvml.utilization()
        if gpu_util is None or gpu_util >= _GPU_UTIL_THRESHOLD:
            return

        now = time.monotonic()
        if now - self._parent._last_bottleneck_mono < _BOTTLENECK_COOLDOWN_S:
            return
        self._parent._last_bottleneck_mono = now

        from vigil.events import TrainingEvent

        event = TrainingEvent(
            project=self._project,
            step=self._step_fn(),
            event_type="dataloader_bottleneck",
            payload={
                "load_time_s": load_time,
                "gpu_utilization": gpu_util,
            },
        )
        self._emitter.emit(event)


class _NvmlHandle:
    """Lazy pynvml wrapper. Silently degrades when pynvml is unavailable."""

    def __init__(self):
        self._handle = None
        self._available: bool | None = None  # None = not yet tried

    def _init(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
            self._available = True
        except Exception:
            self._available = False
        return self._available

    def utilization(self) -> float | None:
        """Return GPU utilization in [0, 1], or None if unavailable."""
        if not self._init():
            return None
        try:
            rates = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            return rates.gpu / 100.0
        except Exception:
            return None
