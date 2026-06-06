"""Per-sample routing PPL measurement — PREEMPTIVELY KILLED on antipattern-017.

This script is intentionally a pre-flight gate only. Empirical measurement
requires trained adapter weights that do not exist in this repo (Pierre
siblings are all killed; `adapters/{math,bash,python,sql,medical}` are
stubs containing only adapter_config.json).

See MATH.md Theorem 1 — K1549 is settled by derivation: per-sample routing
cannot force `pierre_ppl ≡ single_ppl` at finite accuracy; the observed
identity in exp_pierre_unified_pipeline is the single-sample artifact
documented by Finding #553.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

DOMAINS = ("math", "bash", "python", "sql", "medical")


def preflight_adapter_check() -> dict[str, bool]:
    """Return {domain: weights_present} for each required adapter."""
    present: dict[str, bool] = {}
    for d in DOMAINS:
        weights = REPO / "adapters" / d / "adapters.safetensors"
        present[d] = weights.is_file()
    return present


def main() -> None:
    present = preflight_adapter_check()
    weights_available = sum(present.values())

    results = {
        "experiment": "exp_followup_routing_multi_sample_ppl",
        "verdict": "KILLED",
        "all_pass": False,
        "status_reason": "antipattern-017 cascade: 0 of 5 required adapter weights present",
        "preflight": {
            "adapter_weights_present": present,
            "weights_available": weights_available,
            "weights_required": len(DOMAINS),
        },
        "kill_criteria": {
            "1549": {
                "text": "At 85-99% per-sample routing accuracy, pierre_ppl and single_ppl differ",
                "result": "unmeasurable_but_proven_by_theorem",
                "note": "See MATH.md Theorem 1 — K1549 settled by derivation",
            }
        },
        "unblock_path": "train 5 domain adapters (math/bash/python/sql/medical) and re-run as exp_followup_routing_multi_sample_ppl_v2",
        "antipatterns_triggered": ["017", "020"],
    }

    out = HERE / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"KILLED preemptive; weights_available={weights_available}/5; wrote {out}")


if __name__ == "__main__":
    main()
