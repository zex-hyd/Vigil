"""Gradient hook: detects NaN gradients and gradient norm explosions."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from vigil.emitter import Emitter

_DEFAULT_NORM_THRESHOLD = 100.0

# Emit at most one event per (param, event_type) every this many steps.
# Prevents flooding when gradients are consistently exploding.
_COOLDOWN_STEPS = 10


def install(model: "nn.Module", emitter: "Emitter", project: str, step_fn, norm_threshold: float = _DEFAULT_NORM_THRESHOLD) -> None:
    """Register per-parameter gradient hooks on all model parameters."""
    import torch

    # Shared cooldown table: (param_name, event_type) -> last_emitted_step
    cooldown: dict[tuple[str, str], int] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        _register_hook(name, param, emitter, project, step_fn, norm_threshold, torch, cooldown)


def _register_hook(name: str, param, emitter, project, step_fn, norm_threshold, torch, cooldown: dict) -> None:
    def _hook(grad):
        if grad is None:
            return grad

        try:
            step = step_fn()

            if torch.isnan(grad).any():
                _emit_with_cooldown(
                    cooldown=cooldown,
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
                    cooldown=cooldown,
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


def _emit_with_cooldown(cooldown: dict, emitter, project, step: int, event_type: str, param_name: str, grad_norm: float) -> None:
    key = (param_name, event_type)
    last = cooldown.get(key, -_COOLDOWN_STEPS - 1)
    if step - last < _COOLDOWN_STEPS:
        return
    cooldown[key] = step
    _emit_gradient_event(emitter, project, step, event_type, param_name, grad_norm)


def _emit_gradient_event(emitter, project, step: int, event_type: str, param_name: str, grad_norm: float) -> None:
    from vigil.events import TrainingEvent

    event = TrainingEvent(
        project=project,
        step=step,
        event_type=event_type,
        payload={
            "param_name": param_name,
            "grad_norm": grad_norm,
        },
    )
    emitter.emit(event)
