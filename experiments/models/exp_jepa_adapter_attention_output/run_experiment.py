"""run_experiment.py — exp_jepa_adapter_attention_output (PREEMPT-KILL).

This experiment is preempt-killed per Finding #669 (3rd reuse; promotion threshold
confirmed per F#687). No MLX code is written because parent
`exp_jepa_adapter_residual_stream` is PROVISIONAL (F#682) and every KC here
transitively requires target-verified JEPA adapters from the parent.

Secondary block: both KCs (K1848, K1849) are PROXY-only; under F#666 a KC set
without a target-metric gate cannot issue KILL on proxy-FAIL alone. See MATH.md
§1 theorem and §1.1 for both independent preempt reasons.

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
    The verdict is structural: parent target-unverified ⇒ child unidentifiable,
    AND proxy-only KC set violates F#666 target-gating requirement.
    """
    return {
        "experiment_id": "exp_jepa_adapter_attention_output",
        "verdict": "KILLED",
        "kill_reason": "preempt-child-parent-target-unverified",
        "finding_reference": "F#669 (3rd reuse; promotion threshold per F#687 now confirmed)",
        "secondary_block": "F#666 proxy-only KC set (no target-metric gate pre-registered)",
        "parent_experiment": "exp_jepa_adapter_residual_stream",
        "parent_status_at_claim": "provisional",
        "all_pass": False,
        "is_smoke": False,
        "kill_criteria": [
            {
                "id": 1848,
                "text": "Next-embedding MSE on attn_output layers > MSE on residual stream baseline",
                "kind": "proxy",
                "result": "untested",
                "reason": (
                    "preempt-blocked: residual-stream baseline MSE is parent's unverified quantity "
                    "(F#682 provisional). Comparison against untrained reference is vacuous."
                ),
            },
            {
                "id": 1849,
                "text": "SIGReg Epps-Pulley statistic > 0.3 indicating collapse",
                "kind": "proxy",
                "result": "untested",
                "reason": (
                    "preempt-blocked: superficially measurable on a trained attn_output JEPA adapter "
                    "but its 'collapse threshold' semantics require the parent's SIGReg-stability "
                    "claim to be target-validated first (parent K1767/K1769 untested)."
                ),
            },
        ],
        "kc_set_gating": "proxy-only (2 KCs, both proxy; 0 target) — F#666 violation independent of parent status",
        "unblock_condition": (
            "Parent exp_jepa_adapter_residual_stream reaches status=supported with K3 (GSM8K "
            "target accuracy) and K4 (ablation target) SUPPORTED at full scale, AND pre-registration "
            "is augmented with a target-metric KC before re-running (per F#666). Without the latter, "
            "re-run would hit F#666 even if parent is SUPPORTED."
        ),
        "platform_skills_invoked": [
            "/mlx-dev (noted, not used — no code path)",
            "/fast-mlx (noted, not used — no code path)",
        ],
        "base_model": "mlx-community/gemma-4-e4b-it-4bit (per F#627, not loaded)",
        "impl_follow_up_filed": False,
        "impl_follow_up_rationale": (
            "Preempt-structural KILL does not spawn an _impl companion (per F#687 precedent + "
            "reviewer.md §5). Unblock is parent-external: exp_jepa_adapter_residual_stream_impl "
            "already exists at P=3 from parent's PROVISIONAL filing."
        ),
        "notes": (
            "No MLX code was executed. This is a structural preempt-KILL per F#669. "
            "Third reuse of the preempt-child-parent-target-unverified pattern "
            "(F#669 → F#687 → this); promotion threshold hit."
        ),
    }


def main() -> None:
    """Entry point — never raises, always writes results.json."""
    results = build_results()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[preempt-kill] Wrote {out} — verdict=KILLED, reason=preempt F#669 (3rd reuse)")


if __name__ == "__main__":
    main()
