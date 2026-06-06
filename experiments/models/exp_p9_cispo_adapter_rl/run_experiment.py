"""exp_p9_cispo_adapter_rl — preemptive kill, no execution.

MATH.md derives structural impossibility from:
  T1: dep-chain unfulfilled (F#669, 3rd reuse — promote)
  T2: platform-mismatch (F#658 — Unsloth/CISPO are CUDA-only, target is MLX)

No code to run. This stub emits a KILLED results.json for DB/disk consistency.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def main() -> None:
    payload = {
        "experiment_id": "exp_p9_cispo_adapter_rl",
        "verdict": "KILLED",
        "preemptive": True,
        "executed": False,
        "is_smoke": False,
        "all_pass": False,
        "reason": (
            "Preemptive kill — two independent structural impossibilities: "
            "(T1 F#669 3rd-reuse) parent dep exp_p9_unsloth_rl_environment "
            "OPEN and its parent exp_p9_full_stack_integration missing; "
            "(T2 F#658) Unsloth + CISPO are CUDA/Triton-only — no MLX port "
            "exists; target platform is MLX (Apple Silicon, M5 Pro 48GB)."
        ),
        "kill_criteria": {
            "1399": {
                "text": "CISPO adapter > SFT adapter by >= 5pp on GSM8K",
                "predicted": "fail",
                "measured": "not measured",
                "verdict": "FAIL",
            },
            "1400": {
                "text": "CISPO preserves rare-token grads >2x vs PPO",
                "predicted": "fail",
                "measured": "not measured",
                "verdict": "FAIL",
            },
            "1401": {
                "text": "Training stable (no reward hacking / collapse)",
                "predicted": "fail",
                "measured": "not measured",
                "verdict": "FAIL",
            },
        },
        "findings_reused": ["F#658", "F#669", "F#671"],
        "findings_proposed": [
            "F#669 promotion to standalone on 3rd reuse "
            "(sub-axis → axis: inter-experiment-dep-chain-unfulfilled)"
        ],
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
