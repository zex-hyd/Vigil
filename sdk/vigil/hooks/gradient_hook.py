"""Gradient hook: detects NaN gradients and gradient norm explosions."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from vigil.emitter import Emitter

_DEFAULT_NORM_THRESHOLD = 100.0

# Emit at most one event per event_type every this many training steps (global across all params).
# Prevents flooding when many layers exceed the threshold every step.
_COOLDOWN_STEPS = 10


def _session_config_payload() -> dict:
    """Merge ``@watch(config={...})`` into gradient events for the rule engine."""
    try:
        import vigil as _v

        s = _v.current_session()
        if s is None:
            return {}
        cfg = getattr(s, "_config", None)
        return dict(cfg) if cfg else {}
    except Exception:
        return {}


def install(model: "nn.Module", emitter: "Emitter", project: str, step_fn, norm_threshold: float = _DEFAULT_NORM_THRESHOLD) -> None:
    """Register per-parameter gradient hooks on all model parameters."""
    import math

    import torch

    if math.isinf(norm_threshold) or norm_threshold > 1e300:
        return

    # event_type -> last emitted step (shared by all params)
    global_cooldown: dict[str, int] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        _register_hook(name, param, emitter, project, step_fn, norm_threshold, torch, global_cooldown)


def _register_hook(name: str, param, emitter, project, step_fn, norm_threshold, torch, global_cooldown: dict[str, int]) -> None:
    def _hook(grad):
        if grad is None:
            return grad

        try:
            step = step_fn()

            if torch.isnan(grad).any():
                _emit_with_cooldown(
                    global_cooldown=global_cooldown,
                    emitter=emitter,
                    project=project,
                    step=step,
                    event_type="nan_gradient",
                    param_name=name,
                    grad_norm=float("nan"),
                )
                return grad

            norm = grad.detach().norm().item()
            if norm > norm_threshold:
                _emit_with_cooldown(
                    global_cooldown=global_cooldown,
                    emitter=emitter,
                    project=project,
                    step=step,
                    event_type="gradient_explosion",
                    param_name=name,
                    grad_norm=norm,
                )
        except Exception:
            pass

        return grad

    param.register_hook(_hook)


def _emit_with_cooldown(
    global_cooldown: dict[str, int],
    emitter,
    project,
    step: int,
    event_type: str,
    param_name: str,
    grad_norm: float,
) -> None:
    last = global_cooldown.get(event_type, -_COOLDOWN_STEPS - 1)
    if step - last < _COOLDOWN_STEPS:
        return
    global_cooldown[event_type] = step
    _emit_gradient_event(emitter, project, step, event_type, param_name, grad_norm)


def _emit_gradient_event(emitter, project, step: int, event_type: str, param_name: str, grad_norm: float) -> None:
    from vigil.events import TrainingEvent

    payload = dict(_session_config_payload())
    payload["param_name"] = param_name
    payload["grad_norm"] = grad_norm

    event = TrainingEvent(
        project=project,
        step=step,
        event_type=event_type,
        payload=payload,
    )
    emitter.emit(event)
