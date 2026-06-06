"""exp_followup_polar_landing_gemma4 — preempt-structural KILL.

No measurement. F#666-pure standalone (single proxy KC, no target pairing) +
parent-supersession (F#419 gradient-structural V-collapse on Qwen, F#442 joint
Stiefel PoLAR fixes sr=r exactly on Gemma 4, F#444 joint Stiefel scale stability
on Gemma 4) + architecture-irrelevance (Pierre/P1 uses joint Stiefel PoLAR per
F#442/F#444, NOT landing-on-U-only) + disease-vs-symptoms (Qwen-proxy = symptom,
rank-1 single-domain SFT gradient = disease per F#419 Impossibility Structure).

See MATH.md for the formal verdict argument and PAPER.md for the predictions
table. results.json carries the canonical preempt-KILL payload.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

PAYLOAD = {
    "experiment_id": "exp_followup_polar_landing_gemma4",
    "verdict": "KILLED",
    "verdict_reason": "preempt-structural",
    "all_pass": False,
    "is_smoke": False,
    "measurements_taken": 0,
    "kill_criteria": [
        {
            "id": 1561,
            "text": "PoLAR V-collapse reproduces on actual Gemma 4 E4B (Qwen-proxy finding survives port)",
            "result": "untested",
            "reason": "preempt-KILL: F#666-pure standalone (single proxy KC, no target pairing); parent-supersession by F#419 (gradient-structural mechanism universal) + F#442 (joint Stiefel verified on Gemma 4) + F#444 (joint Stiefel scale stability on Gemma 4)",
        }
    ],
    "preempt_basis": {
        "f666_pure_standalone": True,
        "parent_supersession": ["F#419", "F#442", "F#444"],
        "architecture_irrelevance": "Pierre/P1 deploys joint Stiefel PoLAR per F#442/F#444; landing-on-U-only is not in deployment surface",
        "disease_vs_symptoms": "F#419 Impossibility Structure identifies disease as rank-1 single-domain SFT gradient; Qwen-proxy choice is symptom; Gemma 4 port does not change gradient rank",
        "drain_window_index": 32,
        "novel_sub_form": "polar-landing-followup-on-target-when-known-broken-by-parent (1st instance) within audit-2026-04-17+followup-without-rerun super-family (2nd instance, after F#761 spectral-surgery-followup)",
    },
}


def main() -> None:
    out = HERE / "results.json"
    out.write_text(json.dumps(PAYLOAD, indent=2) + "\n")
    print(json.dumps(PAYLOAD, indent=2))


if __name__ == "__main__":
    main()
