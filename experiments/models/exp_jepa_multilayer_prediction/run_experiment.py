"""run_experiment.py — exp_jepa_multilayer_prediction (PREEMPT-KILL).

This experiment is preempt-killed per Finding #669 (5th reuse; promotion
threshold confirmed at F#698 3rd reuse, post-promotion routing). No MLX code
is written because parent `exp_jepa_adapter_residual_stream` is PROVISIONAL
(F#682) and every KC here compares against an L+1 baseline that parent has
not target-validated:

- K1885 (proxy) compares L+2 prediction MSE against L+1 prediction MSE
  (parent K1767, untested).
- K1886 (target) compares L+2 adapter behavioral quality against L+1
  behavioral quality (parent K1768, untested).

KC set IS properly target-gated per F#666 (1 proxy + 1 target), matching
F#699 precedent — no compound F#666 block. Single preempt-block on parent
target-unverification only.

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

    No MLX import or call is made. No model is loaded. No measurement runs.
    The verdict is structural: parent target-unverified ⇒ child unidentifiable.
    """
    return {
        "experiment_id": "exp_jepa_multilayer_prediction",
        "verdict": "KILLED",
        "kill_reason": "preempt-child-parent-target-unverified",
        "finding_reference": "F#669 (5th reuse; promotion-threshold confirmed at F#698, post-promotion routing)",
        "parent_experiment": "exp_jepa_adapter_residual_stream",
        "parent_status_at_claim": "provisional",
        "parent_finding": "F#682 (PROVISIONAL design-only; 4 target-gated KCs untested)",
        "same_parent_repeat_blocker_index": 3,
        "same_parent_prior_children_blocked": [
            "exp_jepa_router_prediction_error (F#687, 2nd F#669 reuse)",
            "exp_jepa_adapter_attention_output (F#698, 3rd F#669 reuse)",
        ],
        "all_pass": False,
        "is_smoke": False,
        "kill_criteria": [
            {
                "id": 1885,
                "text": "L+2 prediction MSE > 2x L+1 prediction MSE (skip-connection doesn't help)",
                "kind": "proxy",
                "result": "untested",
                "reason": (
                    "preempt-blocked: RHS 'L+1 prediction MSE' is parent's unverified "
                    "K1767 claim (step500/step50 loss ratio < 0.5). Comparing against "
                    "an unverified baseline is vacuous: (a) PASS if L+2 trains while "
                    "L+1 is untrained, (b) FAIL if both collapse to noise, (c) "
                    "meaningful only if L+1 MSE itself is target-validated as learning "
                    "real dynamics. 'Skip-gram harder than CBOW' requires CBOW to work "
                    "first."
                ),
            },
            {
                "id": 1886,
                "text": "L+2 trained adapter behavioral quality > 3pp worse than L+1",
                "kind": "target",
                "result": "untested",
                "reason": (
                    "preempt-blocked: RHS 'L+1 behavioral quality' is parent K1768 "
                    "(GSM8K-Hard accuracy >= token-space r=16 LoRA baseline), untested. "
                    "Substituting a different baseline (e.g. direct token-space LoRA "
                    "bypassing L+1 JEPA) would be antipattern-t — KC explicitly says "
                    "'worse than L+1', not 'worse than any baseline'."
                ),
            },
        ],
        "kc_set_gating": "F#666-compliant (1 proxy K1885 + 1 target K1886) — no compound F#666 block, matches F#699 pattern",
        "f666_compound_subcase": False,
        "f669_reuse_index": 5,
        "unblock_condition": (
            "Parent exp_jepa_adapter_residual_stream reaches status=supported via "
            "exp_jepa_adapter_residual_stream_impl (P=1, already filed). Specifically: "
            "K1766 (SIGReg proxy), K1767 (L_pred ratio proxy), K1768 (GSM8K-Hard target), "
            "K1769 (lambda=0 ablation target) all SUPPORTED. No KC-augmentation needed "
            "at re-claim — K1886 already provides target gate per F#666."
        ),
        "platform_skills_invoked": [
            "/mlx-dev (noted, not used — no code path)",
            "/fast-mlx (noted, not used — no code path)",
        ],
        "base_model": "mlx-community/gemma-4-e4b-it-4bit (per F#627, not loaded)",
        "adapter_targets": "v_proj + o_proj (per F#627, not attached)",
        "impl_follow_up_filed": False,
        "impl_follow_up_rationale": (
            "Preempt-structural KILL does not spawn an _impl companion (per F#687/F#698/F#699 "
            "precedent + reviewer.md §5). Unblock is parent-external: "
            "exp_jepa_adapter_residual_stream_impl already exists at P=1 from parent's "
            "PROVISIONAL filing."
        ),
        "notes": (
            "No MLX code was executed. Structural preempt-KILL per F#669. 5th reuse of "
            "the preempt-child-parent-target-unverified pattern (F#669 → F#687 → F#698 → "
            "F#699 → this). Post-promotion routing: F#698 confirmed 3rd-reuse promotion, "
            "F#699 confirmed cross-parent-family application, this 5th reuse is evidence-"
            "of-stability. Same parent as F#687 and F#698 — 3rd child of F#682 PROVISIONAL "
            "to hit preempt-KILL (parent F#682 unblock leverage now ≥3:1)."
        ),
    }


def main() -> None:
    """Entry point — never raises, always writes results.json."""
    results = build_results()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[preempt-kill] Wrote {out} — verdict=KILLED, reason=preempt F#669 (5th reuse)")


if __name__ == "__main__":
    main()
