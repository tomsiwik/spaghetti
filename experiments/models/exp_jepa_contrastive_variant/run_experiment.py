"""run_experiment.py — exp_jepa_contrastive_variant (PREEMPT-KILL, TRIPLE-FIRE).

Preempt-killed before any MLX code executes. Three independent preempt
memories fire simultaneously:

1. F#666-pure standalone — neither KC is a target metric.
   - K1887 next-embedding accuracy: proxy (structural prediction quality).
   - K1888 training stability: safety guard (NaN/divergence detection).
   - No externally-anchored target (task accuracy / behavioral / oracle-gap).

2. §5 tautological-inter-variant-delta — K1887 compares InfoNCE vs MSE
   variants with no external anchor; both are untested realizations of
   parent F#682's JEPA mechanism.

3. F#669 parent-target-unverified (6th reuse) — parent
   exp_jepa_adapter_residual_stream is PROVISIONAL (F#682); K1887's
   MSE-variant RHS is parent's untested mechanism.

Additionally, this is the 4th child of parent F#682 to hit preempt-KILL
(after F#687, F#698, F#727), crossing the same-parent-repeat-blocker
watchlist promotion threshold (per F#727 canonical note).

This scaffold writes a well-formed results.json so downstream tooling
(reviewer, analyst, `experiment complete`) sees a valid artifact. No MLX
import is performed; no model is loaded; no training runs.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    """Return results dict encoding preempt-KILL (triple-fire + promotion trigger).

    No MLX import or call is made. No model is loaded. No measurement runs.
    Three independent structural preempts + a watchlist promotion trigger.
    """
    return {
        "experiment_id": "exp_jepa_contrastive_variant",
        "verdict": "KILLED",
        "kill_reason": "preempt-triple-fire (F#666-pure + §5-tautological + F#669-parent-target-unverified)",
        "triple_fire": True,
        "triple_fire_index": 6,
        "triple_fire_post_promotion": True,
        "triple_fire_memory_anchor": "mem-promotion-triple-fire-mode (anchored F#721)",
        "preempt_memories_fired": [
            {
                "memory": "mem-preempt-f666-pure-standalone",
                "reuse_index": 16,
                "rationale": (
                    "K1887 is proxy (next-embedding accuracy), K1888 is safety-guard "
                    "(NaN detection). Neither is a target metric per guardrail 1007 "
                    "(task accuracy / behavioral / oracle-gap). KC set is "
                    "structurally unsupportable."
                ),
            },
            {
                "memory": "mem-preempt-s5-tautological-inter-variant-delta",
                "reuse_index": 10,
                "rationale": (
                    "K1887 directly compares InfoNCE-variant accuracy to MSE-variant "
                    "accuracy with no external anchor. Both variants are realizations "
                    "of the same untested parent F#682 mechanism; inter-variant delta "
                    "is tautological."
                ),
            },
            {
                "memory": "mem-f669-preempt-parent-target-unverified",
                "reuse_index": 6,
                "rationale": (
                    "Parent exp_jepa_adapter_residual_stream is PROVISIONAL (F#682); "
                    "K1887's 'MSE variant accuracy' RHS is parent's untested K1767/K1768 "
                    "baseline. K1888 tests stability of a loss variant on an unvalidated "
                    "mechanism — stability without behavioral anchor is uninterpretable."
                ),
                "same_parent_repeat_blocker_index": 4,
                "same_parent_prior_children_blocked": [
                    "exp_jepa_router_prediction_error (F#687, 2nd F#669 reuse)",
                    "exp_jepa_adapter_attention_output (F#698, 3rd F#669 reuse)",
                    "exp_jepa_multilayer_prediction (F#727, 5th F#669 reuse)",
                ],
                "watchlist_promotion_triggered": True,
                "promotion_memory_pending": "mem-promotion-same-parent-repeat-blocker (analyst-owned write)",
            },
        ],
        "parent_experiment": "exp_jepa_adapter_residual_stream",
        "parent_status_at_claim": "provisional",
        "parent_finding": "F#682 (PROVISIONAL design-only; 4 target-gated KCs untested)",
        "parent_unblock_leverage": "4:1 (F#687 + F#698 + F#727 + this = 4 children blocked by one parent)",
        "all_pass": False,
        "is_smoke": False,
        "kill_criteria": [
            {
                "id": 1887,
                "text": "InfoNCE variant next-embedding accuracy < MSE variant",
                "kind": "proxy",
                "result": "untested",
                "reason": (
                    "preempt-blocked (triple-fire). "
                    "(a) F#666-pure: no target KC paired with this proxy. "
                    "(b) §5: inter-variant delta with no external anchor. "
                    "(c) F#669: MSE-variant RHS is parent F#682's untested mechanism."
                ),
            },
            {
                "id": 1888,
                "text": "InfoNCE training unstable (loss NaN or divergence > 3 epochs)",
                "kind": "safety-guard",
                "result": "untested",
                "reason": (
                    "preempt-blocked (triple-fire). "
                    "(a) F#666-pure: safety guard is not a target metric (not task "
                    "accuracy, not behavioral, not oracle-gap). "
                    "(c) F#669: stability of a loss variant on an unvalidated JEPA "
                    "mechanism is behaviorally uninformative."
                ),
            },
        ],
        "kc_set_gating": "F#666-pure (no target KC — both KCs are proxy/safety)",
        "f666_compound_subcase": False,
        "f666_pure_subcase": True,
        "f669_reuse_index": 6,
        "s5_reuse_index": 10,
        "same_parent_repeat_blocker_promotion_triggered": True,
        "unblock_condition": (
            "Requires THREE conditions to clear: "
            "(1) KC augmentation adding a target metric (e.g. InfoNCE-variant adapter "
            "GSM8K-Hard accuracy >= LoRA r=16 baseline) — clears F#666-pure AND §5. "
            "(2) Parent exp_jepa_adapter_residual_stream reaches status=supported via "
            "exp_jepa_adapter_residual_stream_impl (P=1, filed) with K1767 + K1768 + "
            "K1769 all SUPPORTED — clears F#669. "
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
            "Preempt-structural KILL does not spawn an _impl companion (per F#687/F#698/"
            "F#699/F#727 precedent + reviewer.md §5). Unblock is (a) KC augmentation "
            "at re-claim (researcher-owned action) and (b) parent's existing "
            "exp_jepa_adapter_residual_stream_impl (P=1)."
        ),
        "memory_promotion_pending": {
            "memory_id": "mem-promotion-same-parent-repeat-blocker",
            "owner": "analyst",
            "trigger": "4th same-parent-F#682 child preempt-KILL",
            "precedent": "F#727 canonical note: 'If 4th same-parent child of F#682 hits preempt-KILL, promote to standalone memory'",
        },
        "notes": (
            "No MLX code executed. Structural preempt-KILL via triple-fire. "
            "Additionally triggers same-parent-repeat-blocker watchlist promotion "
            "(4th F#682 child). Post-promotion F#669 routing (stable across 5 prior "
            "reuses), §5 routing (stable across 9 prior instances), and triple-fire-mode "
            "routing (stable across 5 prior triple-fires) all apply without re-derivation. "
            "Memory-write for same-parent-repeat-blocker promotion is analyst-owned."
        ),
    }


def main() -> None:
    """Entry point — never raises, always writes results.json."""
    results = build_results()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(
        f"[preempt-kill triple-fire] Wrote {out} — verdict=KILLED, "
        f"reason=F#666-pure + §5 + F#669 (6th), same-parent-F#682 child #4 → promotion trigger"
    )


if __name__ == "__main__":
    main()
