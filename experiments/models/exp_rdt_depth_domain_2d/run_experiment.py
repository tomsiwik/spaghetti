"""exp_rdt_depth_domain_2d — preemptive kill, no execution.

Preempt-killed per F#669 (preempt-child-KCs-require-parent-target-claim-unverified).
See MATH.md for the 4 Theorems reducing each KC to parent dep-unfulfillment.

Parents:
  - exp_rdt_loop_lora_gemma4: smoke-PROVISIONAL (F#668). No trained loop-LoRAs.
  - exp_method_composition_k_saturation: KILLED at Phase-1 teacher gate.
    No trained method adapters.

This file exists only for disk-artifact completeness (6/6 doc rule). It writes
results.json with verdict=KILLED, preemptive=true, executed=false.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    outdir = Path(__file__).parent
    results = {
        "experiment_id": "exp_rdt_depth_domain_2d",
        "verdict": "KILLED",
        "preemptive": True,
        "executed": False,
        "is_smoke": False,
        "all_pass": False,
        "reason": (
            "F#669 child-KCs-require-parent-target-claim-unverified. "
            "Parent exp_rdt_loop_lora_gemma4 is smoke-provisional (F#668); "
            "parent exp_method_composition_k_saturation killed at Phase-1 "
            "teacher gate. Neither parent produced trained adapter artifacts "
            "required by K1749-K1752."
        ),
        "kill_criteria": {
            "1749": {
                "pass": False,
                "measured": None,
                "note": "not_measured — T1: requires trained domain × loop adapters",
            },
            "1750": {
                "pass": False,
                "measured": None,
                "note": "not_measured — T2: requires parent target K1740/K1741 curve",
            },
            "1751": {
                "pass": False,
                "measured": None,
                "note": "not_measured — T3: ΔW=0 at init (B=0); requires trained deltas",
            },
            "1752": {
                "pass": False,
                "measured": None,
                "note": "not_measured — T4: Room Model needs trained artifacts (F#571)",
            },
        },
        "findings_reused": ["F#669", "F#668", "F#571", "F#562"],
        "unblock_path": [
            "exp_rdt_loop_lora_gemma4_full (macro follow-up, logged in parent LEARNINGS)",
            "exp_method_composition_k_saturation v2 with Phase-1 teacher gate fix",
        ],
    }
    (outdir / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
