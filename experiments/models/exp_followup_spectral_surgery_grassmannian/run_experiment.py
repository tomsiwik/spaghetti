"""exp_followup_spectral_surgery_grassmannian — preempt-structural KILL.

No measurement. F#666-pure standalone (single proxy KC, no target pairing) +
parent-supersession (F#278 / F#488 / F#64) + architecture-irrelevance
(non-Grassmannian test pool ∉ Pierre deployment surface, which is Grassmannian
by PoLAR construction).

See MATH.md for the formal verdict argument and PAPER.md for the predictions
table. results.json carries the canonical preempt-KILL payload.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

PAYLOAD = {
    "experiment_id": "exp_followup_spectral_surgery_grassmannian",
    "verdict": "KILLED",
    "verdict_reason": "preempt-structural",
    "all_pass": False,
    "is_smoke": False,
    "measurements_taken": 0,
    "kill_criteria": [
        {
            "id": 1560,
            "text": "On non-orthogonal adapter pairs, spectral surgery reduces interference by >=20% vs identity",
            "result": "untested",
            "reason": "preempt-KILL: F#666-pure standalone (single proxy KC, no target pairing)",
        }
    ],
    "preempt_basis": {
        "f666_pure_standalone": True,
        "parent_supersession": ["F#278", "F#488", "F#64"],
        "architecture_irrelevance": "Pierre/P1 uses Grassmannian-orthogonal adapters (PoLAR); non-Grassmannian test pool not in deployment surface",
        "drain_window_index": 31,
        "novel_sub_form": "spectral-surgery-followup-on-irrelevant-test-pool (1st instance)",
    },
}


def main() -> None:
    out = HERE / "results.json"
    out.write_text(json.dumps(PAYLOAD, indent=2) + "\n")
    print(json.dumps(PAYLOAD, indent=2))


if __name__ == "__main__":
    main()
