"""RunMetrics: structured results from arena runs."""

from dataclasses import dataclass, field, asdict


@dataclass
class RunMetrics:
    model_name: str
    param_count: int
    final_loss: float
    val_loss: float | None = None
    learning_speed: int | None = None  # steps to reach loss < threshold
    tokens_per_sec: float = 0.0
    elapsed_s: float = 0.0
    forgetting: dict | None = None  # multi-domain forgetting details

    def to_dict(self) -> dict:
        return asdict(self)


def compute_forgetting(eval_matrix: dict, domains: list[str]) -> dict:
    """Compute forgetting from an eval matrix.

    forgetting[domain] = loss_after_last_phase - loss_after_own_phase
    Positive = forgetting (loss increased).
    """
    phases = list(eval_matrix.keys())
    result = {}
    for i, domain in enumerate(domains):
        own_phase = phases[i]
        last_phase = phases[-1]
        if own_phase == last_phase:
            continue
        before = eval_matrix[own_phase][domain]
        after = eval_matrix[last_phase][domain]
        result[domain] = {
            "after_own": before,
            "after_last": after,
            "forgetting": after - before,
            "pct": 100 * (after - before) / (before + 1e-8),
        }
    return result


def compute_learning_speed(losses: list[float], threshold: float = 2.5) -> int | None:
    """Steps to first reach loss < threshold. None if never reached."""
    for i, l in enumerate(losses):
        if l < threshold:
            return i + 1
    return None
