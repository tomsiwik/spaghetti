"""exp_g4_gs_random_perm_n25 — preempt-structural KILL stub.

No measurement is performed. The verdict is pre-registered KILLED per MATH.md:
F#666-pure-standalone, 9th drain-window instance, 3rd derived-geometric
sub-flavor (stability/perturbation-magnitude semantics), TAXONOMY-REFACTOR
EXECUTION TRIGGER fires (analyst pre-committed option (b) at F#710).

The single pre-registered KC (K1595 "worst/mean <= 1.5x") is a proxy-only
structural-stability ratio of the removal-deviation distribution. No paired
target KC, which under F#666 / guardrail 1007 admits no compliant verdict:
  - proxy-PASS alone  -> tautological support (forbidden by F#666 canonical)
  - proxy-FAIL alone  -> "finding about proxy, not kill" (forbidden)

This script intentionally writes results.json with verdict=KILLED and exits
without loading any model, tokenizer, or dataset. Re-registration path is
described in MATH.md "Unblock path" section.
"""

import json
from pathlib import Path

RESULTS = {
    "verdict": "KILLED",
    "preempt_reason": "F666_PURE_STANDALONE_PROXY_ONLY_KC",
    "sub_flavor": "derived_geometric_stability_perturbation_ratio",
    "sub_flavor_instance_within_derived_geometric": 3,
    "drain_window_instance": 9,
    "taxonomy_refactor_execution_trigger": "FIRES_AT_THIS_INSTANCE",
    "taxonomy_refactor_pre_commit_source": "mem-antipattern-f666-pure-standalone-preempt-kill Escalation block (analyst 2026-04-24)",
    "all_pass": False,
    "is_smoke": False,
    "kill_criteria": [
        {
            "id": 1595,
            "text": "worst/mean <= 1.5x",
            "result": "fail",
            "reason": (
                "proxy-only stability/perturbation-ratio KC violates F#666 / "
                "guardrail 1007 — no paired target-metric KC present; "
                "verdict is structurally tautological regardless of measurement. "
                "Worst/mean removal-deviation ratio is a structural geometric "
                "property of the per-position perturbation distribution, with no "
                "binding to downstream behavioral outcomes (MMLU-Pro, oracle gap, "
                "generation quality)."
            ),
        }
    ],
    "evidence": {
        "parent_finding": 160,
        "parent_structure": (
            "pre-F#666 SUPPORTED 2026-03-28 on 2 proxy-only KCs "
            "(K1 worst/mean ratio < 2.0x; K2 abs worst < 1% at d=256); "
            "zero target-metric KCs. Child stripped K2, kept K1 only — "
            "candidate paired-PROXY-half-strip sub-variant of "
            "mem-antipattern-template-regression."
        ),
        "proxy_only_lineage_inheritance_instance": 2,
        "proxy_only_lineage_inheritance_watchlist_threshold": (
            "MET at this instance (1st was F#710 parent F#72; 2nd is this/F#160). "
            "Per F#704/F#669 convention, 2nd instance triggers watchlist filing."
        ),
        "canonical_f666": 666,
        "guardrail": 1007,
        "drain_window_siblings": [700, 701, 703, 705, 706, 707, 708, 710],
        "planned_refactor_buckets": {
            "derived_geometric": [700, 701, "this"],
            "summary_distributional": [705, 708],
            "detection_classification": [706],
            "routing": [703, 707, 710],
        },
    },
    "platform_skills_invoked": [
        "/mlx-dev (noted, not used — no code path)",
        "/fast-mlx (noted, not used — no code path)",
    ],
    "base_model": (
        "mlx-community/gemma-4-e4b-it-4bit (per F#627, not loaded)"
    ),
    "impl_follow_up_filed": False,
    "impl_follow_up_rationale": (
        "Preempt-structural KILL does NOT spawn an _impl companion "
        "(per F#700–F#710 precedent + reviewer.md §5). Unblock is "
        "pre-registration-external (requires editing DB entry to add a "
        "target KC and restore parent's paired K2), not implementation-external."
    ),
    "notes": (
        "Preempt-structural KILL — no model loaded, no dataset opened, "
        "no tokenizer invoked. See MATH.md for theorem + 5 lemmas + "
        "pre-claim 6-item checklist. Researcher action: file 9th instance, "
        "flag taxonomy-refactor execution trigger to analyst, file 2nd "
        "proxy-only-lineage-inheritance for watchlist promotion, file "
        "candidate paired-PROXY-half-strip as template-regression sub-variant "
        "for analyst classification."
    ),
}


def main() -> int:
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(RESULTS, indent=2))
    print(f"Wrote {out}")
    print(f"Verdict: {RESULTS['verdict']} ({RESULTS['preempt_reason']})")
    print(
        f"Drain-window instance: {RESULTS['drain_window_instance']} — "
        f"taxonomy-refactor execution trigger: "
        f"{RESULTS['taxonomy_refactor_execution_trigger']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
