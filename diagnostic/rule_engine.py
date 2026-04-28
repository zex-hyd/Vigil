"""
Vigil rule engine — deterministic pattern-matching rules, no LLM.

Rules are evaluated in priority order; the first match wins.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from vigil.events import TrainingEvent


@dataclass
class Rule:
    name: str
    condition: Callable[[TrainingEvent, "RuleContext"], bool]
    diagnosis: str
    fix: str | Callable[[TrainingEvent, "RuleContext"], str]
    confidence: float


@dataclass
class RuleContext:
    """Runtime context scraped from the event payload and training config."""
    gradient_accumulation_steps: int = 1
    batch_size: int = 0
    lr_scheduler: str | None = None
    num_workers: int = 0
    loss_spike_ratio: float = 0.0

    @classmethod
    def from_event(cls, event: TrainingEvent) -> "RuleContext":
        """Build context from the event payload. Missing keys use defaults."""
        p = event.payload
        return cls(
            gradient_accumulation_steps=p.get("gradient_accumulation_steps", 1),
            batch_size=p.get("batch_size", 0),
            lr_scheduler=p.get("lr_scheduler", None),
            num_workers=p.get("num_workers", 0),
            loss_spike_ratio=p.get("loss_spike_ratio", 0.0),
        )


# ── Rule definitions ─────────────────────────────────────────────────────────

def _rule_oom_grad_accum() -> Rule:
    def condition(event: TrainingEvent, ctx: RuleContext) -> bool:
        return (
            event.event_type == "oom"
            and ctx.gradient_accumulation_steps == 1
            and ctx.batch_size > 64
        )

    def fix(event: TrainingEvent, ctx: RuleContext) -> str:
        steps = ctx.batch_size // 32
        return f"set gradient_accumulation_steps={steps}"

    return Rule(
        name="oom_grad_accum",
        condition=condition,
        diagnosis="gradient_accumulation disabled, effective batch size too large",
        fix=fix,
        confidence=0.92,
    )


def _rule_nan_loss_no_warmup() -> Rule:
    def condition(event: TrainingEvent, ctx: RuleContext) -> bool:
        return (
            event.event_type == "nan_gradient"
            and ctx.lr_scheduler is None
            and ctx.loss_spike_ratio > 10
        )

    return Rule(
        name="nan_loss_no_warmup",
        condition=condition,
        diagnosis="learning rate too high without warmup, causing gradient explosion",
        fix="add linear warmup for first 500 steps",
        confidence=0.85,
    )


def _rule_dataloader_bottleneck() -> Rule:
    def condition(event: TrainingEvent, ctx: RuleContext) -> bool:
        return (
            event.event_type == "dataloader_bottleneck"
            and ctx.num_workers <= 1
        )

    def fix(event: TrainingEvent, ctx: RuleContext) -> str:
        suggested = max(1, (os.cpu_count() or 2) // 2)
        return f"set num_workers={suggested}"

    return Rule(
        name="dataloader_bottleneck",
        condition=condition,
        diagnosis="dataloader is CPU bottleneck, num_workers too low",
        fix=fix,
        confidence=0.90,
    )


# ── Engine ───────────────────────────────────────────────────────────────────

class RuleEngine:
    def __init__(self):
        self._rules: list[Rule] = [
            _rule_oom_grad_accum(),
            _rule_nan_loss_no_warmup(),
            _rule_dataloader_bottleneck(),
        ]

    def evaluate(self, event: TrainingEvent) -> dict | None:
        """
        Evaluate all rules against the event.

        Returns a diagnosis dict on the first match, or None if no rule fires.
        """
        ctx = RuleContext.from_event(event)

        for rule in self._rules:
            try:
                if rule.condition(event, ctx):
                    fix_str = rule.fix(event, ctx) if callable(rule.fix) else rule.fix
                    return {
                        "rule": rule.name,
                        "diagnosis": rule.diagnosis,
                        "fix": fix_str,
                        "confidence": rule.confidence,
                    }
            except Exception:
                continue

        return None
