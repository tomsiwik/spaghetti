"""run_experiment.py — exp_jepa_frozen_encoder_ablation (PREEMPT-KILL, TRIPLE-FIRE).

Preempt-killed before any MLX code executes. Three independent preempt
memories fire simultaneously:

1. F#666-pure standalone (17th reuse) — the sole KC K1889 is a proxy
   (MSE ratio); no target companion exists. This is |K|=1, maximally
   degenerate: no companion to even re-pair under KC augmentation.

2. §5 tautological-inter-variant-delta (11th reuse) — K1889 compares
   frozen-encoder JEPA MSE vs fine-tuned-encoder JEPA MSE with no
   external anchor; both are untested realizations of parent F#682's
   JEPA mechanism. The 1.5x threshold has no external calibration.

3. F#669 parent-target-unverified (7th reuse) — parent
   exp_jepa_adapter_residual_stream is PROVISIONAL (F#682); K1889's
   fine-tuned-encoder RHS is exactly parent's untested canonical
   training trajectory (K1767).

Additionally, this is the 5th child of parent F#682 to hit preempt-KILL
(after F#687, F#698, F#727, F#728), making it the FIRST post-promotion
instance of mem-promotion-same-parent-repeat-blocker (promoted at F#728
with 4-instance threshold). Expected routing: N+=1 census update only;
no new memory theorem derivation.

This scaffold writes a well-formed results.json so downstream tooling
(reviewer, analyst, `experiment complete`) sees a valid artifact. No MLX
import is performed; no model is loaded; no training runs.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    """Return results dict encoding preempt-KILL (triple-fire + 5th same-parent-F#682).

    No MLX import or call is made. No model is loaded. No measurement runs.
    Three independent structural preempts + 1st post-promotion N+=1 of
    mem-promotion-same-parent-repeat-blocker.
    """
    return {
        "experiment_id": "exp_jepa_frozen_encoder_ablation",
        "verdict": "KILLED",
        "kill_reason": "preempt-triple-fire (F#666-pure + §5-tautological + F#669-parent-target-unverified)",
        "triple_fire": True,
        "triple_fire_index": 7,
        "triple_fire_post_promotion": True,
        "triple_fire_memory_anchor": "mem-promotion-triple-fire-mode (anchored F#721)",
        "preempt_memories_fired": [
            {
                "memory": "mem-preempt-f666-pure-standalone",
                "reuse_index": 17,
                "rationale": (
                    "K1889 is proxy (MSE ratio). No companion KC — |K|=1. "
                    "No target metric per guardrail 1007 (task accuracy / "
                    "behavioral / oracle-gap). KC set structurally "
                    "unsupportable; no companion to re-pair at re-claim."
                ),
            },
            {
                "memory": "mem-preempt-s5-tautological-inter-variant-delta",
                "reuse_index": 11,
                "rationale": (
                    "K1889 directly compares frozen-encoder JEPA MSE to "
                    "fine-tuned-encoder JEPA MSE with no external anchor. "
                    "Both variants are realizations of the same untested "
                    "parent F#682 mechanism. 1.5x threshold arbitrary without "
                    "external calibration."
                ),
            },
            {
                "memory": "mem-f669-preempt-parent-target-unverified",
                "reuse_index": 7,
                "rationale": (
                    "Parent exp_jepa_adapter_residual_stream is PROVISIONAL "
                    "(F#682); K1889's 'fine-tuned encoder JEPA MSE' RHS is "
                    "exactly parent's untested canonical training trajectory "
                    "(parent K1767 measures L_pred on this trajectory). "
                    "Comparing against unverified baseline produces "
                    "unidentifiable sample."
                ),
                "same_parent_repeat_blocker_index": 5,
                "same_parent_post_promotion_instance": 1,
                "same_parent_prior_children_blocked": [
                    "exp_jepa_router_prediction_error (F#687, 2nd F#669 reuse)",
                    "exp_jepa_adapter_attention_output (F#698, 3rd F#669 reuse)",
                    "exp_jepa_multilayer_prediction (F#727, 5th F#669 reuse)",
                    "exp_jepa_contrastive_variant (F#728, 6th F#669 reuse; promotion)",
                ],
                "memory_status": "promoted at F#728; this instance is N+=1 census only (no re-derivation)",
            },
        ],
        "parent_experiment": "exp_jepa_adapter_residual_stream",
        "parent_status_at_claim": "provisional",
        "parent_finding": "F#682 (PROVISIONAL design-only; 4 target-gated KCs untested)",
        "parent_unblock_leverage": "5:1 (F#687 + F#698 + F#727 + F#728 + this = 5 children blocked by one parent)",
        "all_pass": False,
        "is_smoke": False,
        "kill_criteria": [
            {
                "id": 1889,
                "text": "Frozen-encoder JEPA MSE > 1.5x fine-tuned encoder JEPA MSE",
                "kind": "proxy",
                "result": "untested",
                "reason": (
                    "preempt-blocked (triple-fire). "
                    "(a) F#666-pure: no target KC paired with this proxy; |K|=1 "
                    "so no companion to re-pair. "
                    "(b) §5: inter-variant delta with no external anchor; "
                    "1.5x threshold uncalibrated. "
                    "(c) F#669: fine-tuned-encoder RHS is parent F#682's "
                    "untested K1767 training trajectory."
                ),
            },
        ],
        "kc_set_gating": "F#666-pure (|K|=1, sole KC is proxy)",
        "f666_compound_subcase": False,
        "f666_pure_subcase": True,
        "f666_pure_severity": "maximally degenerate — |K|=1, no companion KC exists to re-pair",
        "f669_reuse_index": 7,
        "s5_reuse_index": 11,
        "same_parent_repeat_blocker_post_promotion_n": 5,
        "same_parent_memory_status": "promoted at F#728; post-promotion routing stable",
        "unblock_condition": (
            "Requires THREE conditions to clear: "
            "(1) KC augmentation adding a genuine NEW target metric (not "
            "relabeling — |K|=1 has nothing to re-pair; example: 'Frozen-"
            "encoder JEPA adapter GSM8K-Hard accuracy >= (fine-tuned encoder "
            "- 3pp) at matched param budget on Gemma 4 E4B') — clears F#666-"
            "pure AND §5. "
            "(2) Parent exp_jepa_adapter_residual_stream reaches status="
            "supported via exp_jepa_adapter_residual_stream_impl (P=1, filed) "
            "with K1767 + K1768 + K1769 all SUPPORTED — clears F#669. "
            "(3) Re-claim with augmented KC set and validated parent baseline."
        ),
        "platform_skills_invoked": [
            "/mlx-dev (noted, not used — no code path)",
            "/fast-mlx (noted, not used — no code path)",
        ],
        "base_model": "mlx-community/gemma-4-e4b-it-4bit (per F#627, not loaded)",
        "adapter_targets": "v_proj + o_proj (per F#627, not attached)",
        "impl_follow_up_filed": False,
        "impl_follow_up_rationale": (
            "Preempt-structural KILL does not spawn an _impl companion (per "
            "F#687/F#698/F#699/F#727/F#728 precedent + reviewer.md §5). "
            "Unblock is (a) KC augmentation at re-claim (researcher-owned "
            "action; here requires genuine new target KC, not relabeling) "
            "and (b) parent's existing exp_jepa_adapter_residual_stream_impl "
            "(P=1)."
        ),
        "notes": (
            "No MLX code executed. Structural preempt-KILL via triple-fire. "
            "5th child of parent F#682 to hit preempt-KILL — first post-promotion "
            "instance of mem-promotion-same-parent-repeat-blocker (promoted at "
            "F#728 on 4-child threshold). Post-promotion routing stable across "
            "all three fired memories (F#666-pure, §5, F#669) and the same-"
            "parent memory. |K|=1 distinguishes this from F#728 (|K|=2); "
            "unblock requires adding a target KC from scratch, not relabeling."
        ),
    }


def main() -> None:
    """Entry point — never raises, always writes results.json."""
    results = build_results()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(
        f"[preempt-kill triple-fire] Wrote {out} — verdict=KILLED, "
        f"reason=F#666-pure(17) + §5(11) + F#669(7), same-parent-F#682 child #5 "
        f"(1st post-promotion N+=1 of mem-promotion-same-parent-repeat-blocker)"
    )


if __name__ == "__main__":
    main()
