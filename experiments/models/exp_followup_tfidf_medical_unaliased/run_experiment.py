"""run_experiment.py — exp_followup_tfidf_medical_unaliased (PREEMPT-KILL, F#666-pure, 3rd instance).

This experiment is preempt-killed for a KC-structural violation of guardrail
1007 (Finding #666, target-gated KILL discipline). No MLX code is written
because the pre-registered kill-criterion set consists of a single proxy KC
(K1569 routing weighted accuracy) with no paired target-metric KC. Under
F#666, neither SUPPORTED nor KILLED is derivable from a proxy-only KC set
regardless of empirical outcome.

This is the THIRD F#666-pure standalone instance in the drain window (after
F#700 on `exp_g4_per_layer_cos_baseline` and F#701 on
`exp_adapter_orthogonality_audit`). 3rd instance triggers the escalation
step in `mem-antipattern-f666-pure-standalone-preempt-kill`: analyst should
add an explicit F#666-pure-standalone preempt clause to `reviewer.md §5`.

Distinguished from F#669/F#687/F#698/F#699 (preempt-child-parent-target-
unverified): this experiment has NO parent dependency (`depends_on: []`).
The structural defect is in the KC set itself, not in an upstream artifact.

Distinguished from F#702 hygiene-patch PROVISIONAL: F#702 had target-metric
KCs (wall-clock latency + bitwise-exact token equivalence) and was runnable
under F#666; here K1569 is routing match rate, a pure proxy per F#666
guardrail 1007 enumeration.

This scaffold writes a well-formed `results.json` so downstream tooling
(reviewer, analyst, DB `experiment complete`) sees a valid artifact. No
code path raises; the script always produces a non-empty `results.json`
encoding the preempt-KILL verdict.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    """Return results dict encoding F#666-pure KC-structural preempt-KILL.

    No MLX import or call is made. No TF-IDF classifier is trained. No
    routing accuracy is measured. The verdict is structural: proxy-only KC
    set cannot produce any valid verdict under F#666 regardless of
    empirical outcome.
    """
    return {
        "experiment_id": "exp_followup_tfidf_medical_unaliased",
        "verdict": "KILLED",
        "kill_reason": "kc-structural-f666-pure-proxy-only-3rd-instance",
        "finding_reference": (
            "F#666 (target-gated KILL discipline, guardrail 1007) — "
            "3rd standalone sub-case instance (after F#700, F#701), "
            "orthogonal to F#669 family; triggers reviewer.md §5 "
            "explicit-clause escalation"
        ),
        "parent_experiment": None,
        "parent_status_at_claim": None,
        "all_pass": False,
        "is_smoke": False,
        "kill_criteria": [
            {
                "id": 1569,
                "text": (
                    "Unaliased N=25 TF-IDF routing achieves >=85% weighted "
                    "accuracy (else aliasing was the lift)"
                ),
                "kind": "proxy",
                "result": "untested",
                "reason": (
                    "KC-structural preempt-KILL. Routing weighted accuracy "
                    "is explicitly listed by F#666 guardrail 1007 as a "
                    "forbidden-solo proxy ('classification accuracy, "
                    "routing match rate'). No target-metric KC is paired; "
                    "per F#666 proxy-alone verdicts are tautological "
                    "regardless of pass/fail. F#666 canonical case: 40.2% "
                    "proxy acc + 0.0% target gap proves proxy-PASS alone "
                    "is behaviorally non-predictive."
                ),
            },
        ],
        "kc_set_gating": (
            "F#666-VIOLATING (1 proxy K1569, 0 target). Standalone "
            "F#666-pure case — no parent dependency. Orthogonal to F#698 "
            "which combined parent-unverified (F#669) + F#666 compound. "
            "3rd instance of sub-case (after F#700, F#701) — escalates "
            "to reviewer.md §5 explicit-clause proposal."
        ),
        "secondary_structural_defects": [
            (
                "success_criteria: [] — empty; no SUPPORTED-condition "
                "declared."
            ),
            (
                "references: [] — violates guardrail 1002 (every new "
                "experiment MUST cite an arxiv paper or prior finding). "
                "Notes field cites killed_07.md but the parent "
                "exp_p1_t4_tfidf_routing_v2 is status=SUPPORTED (not "
                "killed); motivation premise is factually wrong."
            ),
        ],
        "platform_correctly_set": (
            "platform=local-apple (not null — this experiment has one "
            "fewer hygiene defect than F#700/F#701)."
        ),
        "prereg_hygiene_antipattern_instance": (
            "2 hygiene defects (vs 3 in F#700/F#701). Below the 3+ "
            "threshold for AP-prereg-hygiene-multi-defect, which "
            "confirms that antipattern keys on hygiene defect *count*, "
            "not on presence of F#666 violation. AP-F666-pure-standalone "
            "applies at any hygiene-defect count (keys on proxy-only "
            "KC structure + standalone)."
        ),
        "unblock_condition": (
            "KC-augmentation required (pre-registration modification): "
            "add a target-metric KC pairing K1569 to a behavioral "
            "outcome (e.g. end-to-end MMLU-Pro subject-domain accuracy "
            "within 3pp of oracle-adapter baseline; or Spearman |r|>=0.4 "
            "between per-sample routing confidence and generation-"
            "quality-delta). Add references: F#666 (violated guardrail), "
            "F#251/F#257 (prior TF-IDF routing), parent "
            "exp_p1_t4_tfidf_routing_v2 (SUPPORTED). Correct notes "
            "field — parent is SUPPORTED not killed; 'self-inflicted "
            "break' premise is factually wrong. Populate success_criteria "
            "mirroring KC pass condition. Platform already set. "
            "Post-claim KC mutation is antipattern-u — edits must happen "
            "before re-claim. Alternative (recommended): close this "
            "pre-reg as structurally-malformed and re-register with "
            "target-metric pair and corrected motivation."
        ),
        "platform_skills_invoked": [
            "/mlx-dev (noted, not used — no code path)",
            "/fast-mlx (noted, not used — no code path)",
        ],
        "base_model": (
            "mlx-community/gemma-4-e4b-it-4bit (per F#627, not loaded)"
        ),
        "prior_experiment_not_rerun": (
            "exp_p1_t4_tfidf_routing_v2 (status=SUPPORTED, K1238 PASS: "
            "N=25 weighted acc 84.2% with hard-negative clinical_knowledge). "
            "Parent already does not alias medical↔clinical_knowledge; "
            "the follow-up's premise 'aliasing was the lift' is "
            "factually wrong. Not re-run."
        ),
        "impl_follow_up_filed": False,
        "impl_follow_up_rationale": (
            "Preempt-structural KILL does NOT spawn an _impl companion "
            "(per F#687/F#698/F#699/F#700/F#701 precedent + "
            "reviewer.md §5). Unblock is pre-registration-external "
            "(requires editing DB entry to add a target KC + fix "
            "motivation), not implementation-external."
        ),
        "drain_subcase_taxonomy": {
            "sub_case": "F#666-pure standalone KC-structural preempt-KILL",
            "orthogonal_to": "F#669 family (parent-unverified)",
            "occurrence_index": 3,
            "prior_instances": [
                "F#700 (exp_g4_per_layer_cos_baseline)",
                "F#701 (exp_adapter_orthogonality_audit)",
            ],
            "escalation_trigger": (
                "3rd instance reached. Per "
                "mem-antipattern-f666-pure-standalone-preempt-kill "
                "escalation rule, analyst should add explicit "
                "F#666-pure-standalone preempt clause to reviewer.md "
                "§5 (currently only F#669 family is explicit)."
            ),
        },
        "semantic_corroboration": (
            "Parent exp_p1_t4_tfidf_routing_v2 is status=SUPPORTED with "
            "N=25 weighted acc 84.2% on disjoint splits with hard-"
            "negative clinical_knowledge. The 85% threshold of K1569 "
            "is +0.8pp above the parent's measured value — a marginal "
            "proxy delta, not a behavioral finding. The follow-up's "
            "premise ('aliasing was the lift') is also factually wrong: "
            "the parent already does not alias medical↔clinical_"
            "knowledge. F#666 canonical result further confirms that "
            "proxy accuracy is not behaviorally predictive (40.2% "
            "proxy + 0.0% target gap)."
        ),
        "notes": (
            "No MLX code was executed. This is a KC-structural preempt-"
            "KILL under F#666 (guardrail 1007), independent of any "
            "parent dependency. The experiment has no parent "
            "(depends_on: []) but the KC set is malformed: 1 proxy KC "
            "(routing weighted accuracy) with no target pairing. Under "
            "F#666, no outcome can produce a valid verdict. 3rd such "
            "instance in drain window (after F#700, F#701) → escalates "
            "to reviewer.md §5 explicit clause (analyst action)."
        ),
    }


def main() -> None:
    """Entry point — never raises, always writes results.json."""
    results = build_results()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(
        f"[preempt-kill] Wrote {out} — verdict=KILLED, "
        "reason=F#666-pure KC-structural (proxy-only, no target pairing), "
        "3rd drain-window instance"
    )


if __name__ == "__main__":
    main()
