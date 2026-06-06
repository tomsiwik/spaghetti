"""run_experiment.py — exp_jepa_router_prediction_error (PREEMPT-KILL).

This experiment is preempt-killed per Finding #669. No MLX code is written
because parent `exp_jepa_adapter_residual_stream` is PROVISIONAL (F#682) and
every KC here transitively requires target-verified JEPA adapters from the
parent. See MATH.md §1 for the theorem.

This scaffold writes a well-formed `results.json` so downstream tooling
(reviewer, analyst, DB `experiment complete`) sees a valid artifact. No code
path raises: the script always produces a non-empty `results.json` that
encodes the preempt-kill verdict and structurally-untestable KCs.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    """Return results dict encoding preempt-KILL.

    No MLX import or call is made. No model is loaded. No training runs.
    The verdict is structural: parent target-unverified ⇒ child unidentifiable.
    """
    return {
        "experiment_id": "exp_jepa_router_prediction_error",
        "verdict": "KILLED",
        "kill_reason": "preempt-child-parent-target-unverified",
        "finding_reference": "F#669",
        "parent_experiment": "exp_jepa_adapter_residual_stream",
        "parent_status_at_claim": "provisional",
        "all_pass": False,
        "is_smoke": False,
        "kill_criteria": [
            {
                "id": 1775,
                "text": "K1 proxy: routing agreement with classification baseline > 70% on N=25 held-out prompts",
                "result": "untested",
                "reason": "preempt-blocked: parent JEPA adapters do not exist (F#682 provisional)",
            },
            {
                "id": 1776,
                "text": "K2 target: task accuracy under JEPA routing >= oracle, |Delta_oracle_gap| < 2pp at N=25",
                "result": "untested",
                "reason": "preempt-blocked: parent target KCs K3/K4 untested (F#682)",
            },
            {
                "id": 1777,
                "text": "K3 target: beats softmax classification router by >= 5pp absolute task accuracy, n>=200",
                "result": "untested",
                "reason": "preempt-blocked: requires parent-trained predictors with verified dynamics",
            },
            {
                "id": 1778,
                "text": "K4 serving-cost: per-token routing latency < 1.2x single adapter forward pass",
                "result": "untested",
                "reason": "preempt-blocked: latency of degenerate predictor is uninterpretable as KC signal",
            },
        ],
        "unblock_condition": (
            "Parent exp_jepa_adapter_residual_stream reaches status=supported "
            "with K3 (GSM8K target accuracy) and K4 (ablation target) SUPPORTED at full scale. "
            "Then re-claim this child with parent target-validated JEPA adapters."
        ),
        "platform_skills_invoked": ["/mlx-dev (noted, not used — no code path)", "/fast-mlx (noted, not used)"],
        "base_model": "mlx-community/gemma-4-e4b-it-4bit (per F#627, not loaded)",
        "notes": "No MLX code was executed. This is a structural preempt-KILL per F#669.",
    }


def main() -> None:
    """Entry point — never raises, always writes results.json."""
    results = build_results()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[preempt-kill] Wrote {out} — verdict=KILLED, reason=preempt F#669")


if __name__ == "__main__":
    main()
