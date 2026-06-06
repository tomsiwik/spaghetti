"""exp_composition_runtime_vs_merge_n10 — preempt-KILL stub.

No execution: see MATH.md §1-§4.
- K1894 (runtime > 2x merge latency) is structurally FALSE by F#399 BW-bound
  theorem: pre-merge speedup ceiling < 1.1x at rank*N = 80 << d_model.
- K1895 (merge quality > 5pp worse) is branch-redundant: F#66 (bf16 50% delta
  loss), F#510/F#511 (standard-LoRA destroys), F#543 (uniform N=5 2.57x bloat),
  F#406/F#54 (Grassmannian runtime SUPPORTED). Every plausible branch has a
  published answer.
- Both KCs additionally F#666-pure under guardrail 1007 (K1895 unbound
  "quality"; K1894 infrastructure-benchmark per F#715 bucket).

2nd drain-window instance of method-dependent redundancy (F#731 was 1st) =>
PROMOTION trigger for standalone memory.
"""

from __future__ import annotations

import json
import pathlib


def main() -> None:
    out = {
        "verdict": "KILLED",
        "reason": (
            "preempt-structural: K1894 FAIL by F#399 BW-bound theorem "
            "(pre-merge speedup ceiling < 1.1x at practical rank); "
            "K1895 method-dependent redundancy (F#66 bf16 50% delta loss, "
            "F#510/F#511 standard-LoRA destroys, F#543 uniform N=5 2.57x bloat, "
            "F#406/F#54 Grassmannian runtime SUPPORTED) -- every branch covered; "
            "both KCs F#666-pure (K1895 no dataset, K1894 infrastructure per F#715). "
            "2nd method-dependent-redundancy instance => PROMOTION"
        ),
        "all_pass": False,
        "k1894_runtime_gt_2x_merge_latency": "inconclusive (structurally FALSE per Thm 1 / F#399)",
        "k1895_merge_quality_gt_5pp_worse": "inconclusive (branch-redundant per Thm 2; target-unbound per Thm 3)",
        "references": [
            "F#399",
            "F#66",
            "F#406",
            "F#54",
            "F#510",
            "F#511",
            "F#543",
            "F#715",
            "F#731",
        ],
        "is_smoke": False,
        "preempt_kill": True,
        "method_dependent_redundancy_instance": 2,
        "promotion_trigger": True,
    }
    out_path = pathlib.Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
