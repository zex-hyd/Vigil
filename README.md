# Vigil

> ML training observability — like Sentry, but for PyTorch failures.

Add **2 lines** to your training script and get automatic diagnosis of OOM crashes, gradient explosions, and dataloader bottlenecks.

```python
import vigil

@vigil.watch(project="my-project")
def train():
    model = MyModel()
    vigil.watch_model(model)                    # gradient hooks
    loader = vigil.watch_dataloader(loader)     # dataloader profiler

    for epoch in range(100):
        for batch in loader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            vigil.step()
```

When a failure occurs, Vigil prints a structured diagnosis:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Vigil  |  my-project  |  step 312
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Error:   oom
Cause:   gradient_accumulation disabled, effective batch size too large
Fix:     set gradient_accumulation_steps=4
Confidence: 92%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Project layout

```
vigil/
  sdk/
    vigil/
      __init__.py          # @watch decorator, public API
      events.py            # TrainingEvent dataclass
      emitter.py           # Non-blocking async queue + 500ms flush loop
      hooks/
        cuda_hook.py       # CUDA OOM interception + memory snapshot
        gradient_hook.py   # NaN / explosion detection via param.register_hook
        dataloader_profiler.py  # Batch load timing + pynvml GPU utilisation
    setup.py
  diagnostic/
    rule_engine.py         # Deterministic rule matching (no LLM)
  tests/
    test_oom.py            # 7 tests covering OOM, rules, emitter, integration
```

## Installation

```bash
cd sdk
pip install -e .
```

## Running tests

```bash
python tests/test_oom.py
# or
pytest tests/ -v
```

## Hooks

| Hook | Trigger | Event type |
|------|---------|------------|
| CUDA OOM | `torch.cuda.OutOfMemoryError` | `oom` |
| Gradient | NaN detected | `nan_gradient` |
| Gradient | norm > threshold (default 100) | `gradient_explosion` |
| Dataloader | GPU util < 40% AND load time > 100ms | `dataloader_bottleneck` |

## Rules (deterministic, no LLM)

| Rule | Condition | Confidence |
|------|-----------|------------|
| `oom_grad_accum` | OOM + batch_size > 64 + grad_accum == 1 | 92% |
| `nan_loss_no_warmup` | NaN gradient + no LR scheduler + loss spike > 10× | 85% |
| `dataloader_bottleneck` | Bottleneck event + num_workers ≤ 1 | 90% |

## Design constraints

- **Never blocks the training loop** — all hooks call `queue.put_nowait()` only
- **< 0.5% overhead target** — gradient hooks use in-process tensor ops, no subprocess
- **GPU metrics via pynvml** — no `nvidia-smi` subprocess
- **Python 3.9+, PyTorch 2.0+**

**Note:** Overhead benchmarks should use a **fixed seed**, **multiple iterations**, and report a **median** (or trimmed mean). A **single Colab run** often shows **±2% noise** due to GPU scheduling and shared-runtime variance — that is normal and not a regression by itself.

**Dataloader bottleneck events** are rate-limited to **at most one emit per 30 seconds** per wrapped `DataLoader` (avoids flooding the terminal when every batch qualifies).
