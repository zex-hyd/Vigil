"""Async event emitter: non-blocking enqueue + background daemon flush every 500ms."""
from __future__ import annotations

import json
import queue
import sys
import threading
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vigil.events import TrainingEvent

_FLUSH_INTERVAL_S = 0.5
_MAX_QUEUE = 4096


class Emitter:
    def __init__(self, on_events=None):
        self._q: queue.Queue[TrainingEvent] = queue.Queue(maxsize=_MAX_QUEUE)
        # on_events: optional callable that receives a list of TrainingEvent
        self._on_events = on_events
        self._thread = threading.Thread(target=self._flush_loop, daemon=True, name="vigil-emitter")
        self._stop = threading.Event()
        self._thread.start()

    def emit(self, event: "TrainingEvent") -> None:
        """Enqueue an event. Never blocks — drops if queue full."""
        try:
            self._q.put_nowait(event)
        except queue.Full:
            pass

    def _flush_loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(timeout=_FLUSH_INTERVAL_S)
            self._drain()

    def _drain(self) -> None:
        batch: list[TrainingEvent] = []
        try:
            while True:
                batch.append(self._q.get_nowait())
        except queue.Empty:
            pass

        if not batch:
            return

        if self._on_events:
            try:
                self._on_events(batch)
            except Exception:
                pass
        else:
            for event in batch:
                self._default_sink(event)

    def _default_sink(self, event: "TrainingEvent") -> None:
        """Serialize to JSON and write to stdout (gRPC transport replaces this later)."""
        try:
            print(json.dumps(asdict(event)), flush=True, file=sys.stdout)
        except Exception:
            pass

    def flush_now(self) -> None:
        """Deliver queued events synchronously on the calling thread.

        Jupyter/Colab sometimes does not capture stdout from daemon threads — call this
        after fatal events (e.g. CUDA OOM) so diagnostics appear in the cell output.
        """
        self._drain()

    def shutdown(self, wait: bool = True) -> None:
        self._stop.set()
        if wait:
            self._thread.join(timeout=2.0)
        self._drain()


# Module-level singleton, replaced by Vigil.__init__ with a configured instance.
_default_emitter: Emitter | None = None


def get_emitter() -> Emitter:
    global _default_emitter
    if _default_emitter is None:
        _default_emitter = Emitter()
    return _default_emitter


def set_emitter(emitter: Emitter) -> None:
    global _default_emitter
    _default_emitter = emitter
