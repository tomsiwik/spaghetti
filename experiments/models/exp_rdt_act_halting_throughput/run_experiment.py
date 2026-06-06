"""Preemptive-kill stub for exp_rdt_act_halting_throughput.

No code executed. Parent dependency (exp_rdt_loop_lora_gemma4) is
smoke-provisional; its target KCs K1740/K1741/K1742 are untested. All four
child KCs (K1745-K1748) structurally require parent target claim SUPPORTED.

See MATH.md Theorem 1-2 and PAPER.md for the full argument.

To run: wait for exp_rdt_loop_lora_gemma4_full (macro P1) to be SUPPORTED,
then redesign this experiment with access to a trained loop-LoRA Gemma 4
checkpoint.
"""
import json
import sys
from pathlib import Path


def main():
    out = Path(__file__).parent / "results.json"
    payload = {
        "verdict": "KILLED",
        "preemptive": True,
        "executed": False,
        "is_smoke": False,
        "all_pass": False,
        "elapsed_sec": 0.0,
        "reason": (
            "dependency-unfulfilled: parent exp_rdt_loop_lora_gemma4 is "
            "smoke-provisional; K1740/K1741/K1742 untested. Child KCs "
            "K1745-K1748 transitively require parent target claim SUPPORTED."
        ),
        "kill_criteria": {
            "K1745": "not_measurable_without_trained_loop_lora",
            "K1746": "not_measurable_without_trained_loop_lora",
            "K1747": "not_measurable_without_trained_loop_lora",
            "K1748": "not_measurable_without_trained_loop_lora",
        },
        "antipatterns_flagged": [
            "preempt-child-KCs-require-parent-target-claim-unverified",
        ],
        "precedent_findings": ["F#513", "F#558"],
        "unblock_path": "queue exp_rdt_loop_lora_gemma4_full (macro P1); rerun after parent K1740 or K1742 SUPPORTED at full scale",
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"PREEMPTIVE-KILL: wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
