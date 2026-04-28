from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class TrainingEvent:
    project: str
    step: int
    event_type: str  # "oom" | "nan_gradient" | "gradient_explosion" | "dataloader_bottleneck"
    payload: dict
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_ns: int = field(default_factory=time.time_ns)
