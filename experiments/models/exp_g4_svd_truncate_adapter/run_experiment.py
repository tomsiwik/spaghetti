"""exp_g4_svd_truncate_adapter — PREEMPT-KILL stub.

Structural preempt-KILL on the tautological-inter-variant-delta antipattern
(§5 clause, 4th instance, sub-variant intra-adapter-rank-delta). No measurement
performed. Writes results.json with verdict=KILLED and documents the structural
lemma chain. See MATH.md for the full proof.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    exp_dir = Path(__file__).parent
    results = {
        "experiment_id": "exp_g4_svd_truncate_adapter",
        "verdict": "KILLED",
        "all_pass": False,
        "is_smoke": False,
        "preempt_structural": True,
        "antipattern": "tautological-inter-variant-delta-ignores-base-baseline",
        "antipattern_instance": 4,
        "antipattern_sub_variant": "intra-adapter-rank-delta",
        "clause": "reviewer.md §5 (promoted at F#709, 3rd instance)",
        "kill_criteria": [
            {
                "id": 1611,
                "text": "r=4 within 5% of r=6 on MMLU-Pro",
                "result": "UNMEASURABLE",
                "note": (
                    "Comparison-only KC without per-variant base-anchor. "
                    "Parent F#477 established r=6 q_proj Gemma 4 K1226 FAIL "
                    "(adapted_acc=0.480 < 0.50). Degenerate-equivalence regime: "
                    "both M_4 and M_6 collapse to M_base, |M_4-M_6| <= 0.05 "
                    "trivially with zero SVD-truncation quality mechanism."
                ),
            }
        ],
        "lemmas": [
            "L1 degenerate-equivalence trivially satisfies K1611",
            "L2 F#166 prerequisite-gate unmet (no per-variant base-anchor)",
            "L3 F#477 parent-target-FAIL inheritance on MCQ regime",
            "L4 intra-adapter-rank-delta = 4th sub-variant of §5 antipattern",
            "L5 hygiene-independence (2 defects below 3+ threshold)",
        ],
        "unblock_path": (
            "v2 exp_g4_svd_truncate_adapter_v2_domain_ppl: restrict to 3 trained "
            "domains (code/math/medical), use F#325 PPL-ratio metric with "
            "per-variant base-anchor floor (PPL_r < PPL_base) AND pair delta "
            "(PPL_4/PPL_6 <= 1.05)."
        ),
        "references": [
            "F#477 (killed 2026-04-11)",
            "F#627 (supported 2026-04-19)",
            "F#325 (supported 2026-04-06)",
            "F#666 (conclusive 2026-04-19)",
            "F#704 (killed 2026-04-24)",
            "F#709 (killed 2026-04-24)",
            "F#166 (prerequisite gate)",
        ],
    }
    out = exp_dir / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
