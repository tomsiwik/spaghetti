"""exp_g4_routing_family_equivalence — PREEMPTIVE KILL stub.

Not executed. See MATH.md for the five-lemma proof that K1584 is either
degenerate-equivalence (L1), prerequisite-gate unmet (L2),
base-beat-structurally-unlikely (L3), parent-caveat-inheritance-failure (L4,
3rd instance of template-regression), or 3rd instance of
tautological-inter-adapter-delta-ignores-base-baseline triggering §5
reviewer.md promotion (L5).

No runnable code. Running this file writes results.json and exits 0.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

RESULTS = {
    "experiment_id": "exp_g4_routing_family_equivalence",
    "status": "KILLED",
    "verdict": "KILLED",
    "preemptive": True,
    "executed": False,
    "is_smoke": False,
    "all_pass": False,
    "preempt_reason": "TAUTOLOGICAL_INTER_ADAPTER_DELTA_IGNORES_BASE_BASELINE_3RD_INSTANCE",
    "kill_criteria": [
        {
            "id": 1584,
            "text": "max pairwise gap < 2pp at N=5",
            "result": "untested",
            "rationale": (
                "Structurally uninformative KC. 3rd instance of tautological-inter-adapter-delta-"
                "ignores-base-baseline (1st K1552, 2nd K1577/F#704). See MATH.md §1 L1-L5."
            ),
        },
    ],
    "findings_reused": [
        {"id": 150, "use": "Parent finding — explicit caveat: quality comparison vacuous under zero expert specialization (4-decimal-place equivalence); child inherits flagged structure"},
        {"id": 165, "use": "OS-top2 killed on Falcon-E-3B; NTP adapter degradation; anchors L2 prerequisite-gate reasoning"},
        {"id": 166, "use": "Prerequisite gate: single adapter must beat base before composition is testable"},
        {"id": 167, "use": "Runtime LoRA IS output-space MoE; binding constraint is per-adapter base quality"},
        {"id": 168, "use": "Cross-terms structurally impossible in output-space LoRA composition (LoRI)"},
        {"id": 477, "use": "Gemma 4 rank-6 single-adapter base-beat rate 2/5 domains; shared-failure regime makes G<2pp low-info"},
        {"id": 704, "use": "2nd instance of tautological-inter-adapter-delta; promotion threshold reached per repo convention"},
        {"id": 705, "use": "1st template-regression sub-variant (stale-caveat-inheritance)"},
        {"id": 708, "use": "2nd template-regression sub-variant (paired-design-half-stripping); watchlist memory filed"},
    ],
    "sibling_precedents": [
        {
            "experiment_id": "exp_followup_output_space_qa_adapters",
            "kc_id": 1552,
            "status": "killed",
            "killed_on": "2026-04-19",
            "relation": "1st-instance-of-antipattern",
            "delta_direction": ">=5pp (large, tautological by format-incompatibility)",
        },
        {
            "experiment_id": "exp_followup_routing_output_space_top2",
            "kc_id": 1577,
            "finding_id": 704,
            "status": "killed",
            "killed_on": "2026-04-24",
            "relation": "2nd-instance-of-antipattern; promotion threshold reached",
            "delta_direction": ">=5pp (large, tautological by format-incompatibility)",
        },
    ],
    "antipattern_flags": [
        "tautological-inter-adapter-delta-ignores-base-baseline-3rd-instance-promote-to-reviewer-md-section-5",
        "template-regression-explicit-anti-caveat-inheritance-3rd-instance-promote-to-formal-antipattern",
        "prerequisite-gate-unmet-routing-composition",
        "degenerate-equivalence-branch-gap-by-shared-failure",
        "parent-caveat-inheritance-failure-from-finding-150",
    ],
    "no_rerun_justification": (
        "A valid v2 requires: (a) pre-registered base-anchored quality gate per variant; "
        "(b) replace inter-variant delta with per-variant base-beat KCs; (c) decouple "
        "'verified Grassmannian' from KC set — orthogonality as fixture, not confound; "
        "(d) cite F#167/F#168 — binding constraint is per-adapter base quality. "
        "Requires a new experiment ID, not a re-run."
    ),
    "behavioral_predictions_not_measured": {
        "Q_variant_MMLU_Pro_mean": "approximately equal to base across all 4 variants (F#150: identical to 4 decimal places under orthogonality)",
        "G_pairwise_gap": "< 0.0001 (vacuously < 0.02 per F#150 caveat)",
        "single_adapter_base_beat_rate": "approximately 2/5 domains (F#477 inheritance)",
        "mean_Q_variant_minus_Q_base": "in [-0.03, +0.02] (shared shallow regime, F#477)",
        "thesis_relevance_of_PASS": "zero — PASS compatible with all-variants-fail-to-beat-base",
    },
    "promotion_actions_for_analyst": {
        "promote_antipattern_to_reviewer_md_section_5": (
            "mem-antipattern-tautological-inter-adapter-delta-ignores-base-baseline; "
            "3-instance-promote convention triggered (1st K1552, 2nd K1577/F#704, 3rd K1584)"
        ),
        "promote_template_regression_watchlist_to_formal_antipattern": (
            "mem-watchlist-f666pure-template-regression; 3rd sub-variant instance "
            "(F#705 stale-caveat, F#708 paired-design-half-strip, F#709 explicit-anti-caveat)"
        ),
        "extend_clause_to_cover_both_delta_directions": (
            ">= and < both admit degenerate PASS; tautology is direction-symmetric"
        ),
    },
}


def main() -> None:
    (HERE / "results.json").write_text(json.dumps(RESULTS, indent=2) + "\n")
    print("Preemptive kill — no run. See MATH.md and results.json.")


if __name__ == "__main__":
    main()
