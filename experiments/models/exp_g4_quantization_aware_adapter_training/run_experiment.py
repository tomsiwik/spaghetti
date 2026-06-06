"""Refusal scaffold — exp_g4_quantization_aware_adapter_training.

Structural preempt-KILL under three independent blockers (any one sufficient):
  1. F#666-pure-standalone — both KCs are non-target metrics
       K1920 = PPL gap (proxy)
       K1921 = training-time ratio (meta-engineering, not a metric)
  2. F#502/F#646 — success_criteria=[] in DB; cohort instance
  3. Self-documented predicate-not-met — DB notes (frozen by prior researcher
     2026-04-25 release) require: (a) lock-in of QAT-LoRA paper reference and
     (b) derivation of STE composition with mlx.QuantizedLinear (no direct
     gradient path; forward replacement, not wrapping). Neither is satisfied.

This file does not load any model, does not invoke /mlx-dev or /fast-mlx,
does not write any training-loop code, and emits a refusal results.json so
that reviewer can issue
  experiment update --status killed
with finding-add SKIPPED per F#769 closing-note (ledger-explosion antipattern
at Nth instance of established cohort).

See MATH.md §2 for the three independent blockers and §5 for the antipattern
scan.
"""

from __future__ import annotations

import json
import pathlib
import sys

EXPERIMENT_ID = "exp_g4_quantization_aware_adapter_training"
RESULTS_PATH = pathlib.Path(__file__).parent / "results.json"


def main() -> int:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "verdict": "KILLED",
        "all_pass": False,
        "is_smoke": False,
        "kill_criteria": {
            "K1920": {
                "text": "QAT adapter PPL within 0.05 of full-precision adapter",
                "result": "untested",
                "reason": (
                    "Proxy metric (perplexity gap) with no paired target-metric KC."
                    " Per F#666 (guardrail 1007), proxy KC requires a target KC for"
                    " kill or support. Per CLAUDE memory measured r~=0.08 between PPL"
                    " and behavioral task quality, PPL gap alone cannot decide whether"
                    " QAT preserves adapter behavior."
                ),
            },
            "K1921": {
                "text": "QAT training time > 2x standard LoRA training time",
                "result": "untested",
                "reason": (
                    "Meta-engineering metric (wall-clock ratio). Neither a proxy for"
                    " behavioral quality nor a target-metric for the scientific claim"
                    " 'QAT preserves adapter behavior'. It is a budget gate, not a"
                    " kill-criterion."
                ),
            },
        },
        "preempt_kill": {
            "form": "structural (three independent blockers)",
            "blocker_1_f666_pure_standalone": {
                "violation": "no proxy/target KC pairing",
                "canonical_instances": ["F#700", "F#705", "F#706", "F#707", "F#722"],
                "guardrail": "1007 target-gated KILL",
            },
            "blocker_2_f502_f646_schema_incomplete": {
                "violation": "success_criteria=[] in DB (verified via experiment get)",
                "db_warning": "INCOMPLETE: success_criteria, references, platform, experiment_dir, kill_results",
                "canonical_instances": ["F#629", "F#655", "F#769"],
                "plan_md_block": "PLAN.md §1: success_criteria=[] blocks supported verdict",
            },
            "blocker_3_predicate_not_met": {
                "violation": "self-documented in DB notes (prior researcher release)",
                "predicate_a_citation_lock": (
                    "no specific QAT-LoRA paper reference in DB references=[];"
                    " arxiv:2402.10193 (LoftQ) and arxiv:2310.08659 are SUGGESTED"
                    " by prior researcher but not LOCKED as KC reference"
                ),
                "predicate_b_mechanism_lock": (
                    "no derivation of STE composition with mlx.QuantizedLinear;"
                    " forward must be REPLACED (not wrapped) because the dequant"
                    " op has no native grad path in MLX 0.31"
                ),
                "guardrail_1002": "every new experiment MUST cite arxiv or prior finding",
            },
            "ledger_explosion_guard": (
                "F#769 closing-note: SKIP finding-add at Nth instance of established"
                " cohort. Each of the 3 blockers above is a closed/established cohort."
            ),
            "finding_add": "SKIP",
            "reviewer_evidence": (
                "F#666-pure-standalone (no proxy/target pair) + F#502/F#646 schema"
                " (success_criteria=[]) + self-documented predicate-not-met"
                " (citation + STE-MLX mechanism). Triple-fire structural KILL."
            ),
        },
        "skills_invoked": {
            "mlx_dev": False,
            "fast_mlx": False,
            "rationale": (
                "no platform code written; refusal scaffold only; skill invocation"
                " deferred to unblocked iteration when predicates §2.3 are resolved"
            ),
        },
        "doom_loop_check": {
            "prior_iteration_verdict": "KILLED (preempt-structural, PROD F#765 no-parent sub-form)",
            "this_iteration_verdict": "KILLED (preempt-structural, F#666-pure + F#502 + predicate)",
            "structurally_different_mechanism": True,
            "rationale": (
                "two consecutive KILLs but on distinct mechanisms (PROD-deliverable"
                " vs F#666-pure standalone). Not A->B->A->B alternation. doom_loop.py"
                " exit=0."
            ),
        },
        "released_p2_to_p4_history": {
            "prior_action": "researcher 2026-04-25 lowered priority 2->4 with explicit notes",
            "this_iteration_action": (
                "preempt-KILL rather than further release; the predicate-not-met"
                " condition will not resolve by re-claiming at lower priority — it"
                " requires upstream KC rewrite (paired target metric) and a citation"
                " RFC, neither of which is researcher-scope work"
            ),
        },
        "reviewer_route": {
            "expected_action": "experiment update --status killed",
            "expected_evidence": (
                "F#666-pure-standalone + F#502/F#646 + predicate-not-met (triple-fire);"
                " F#769 ledger-explosion: no new finding"
            ),
            "finding_add": "SKIP",
        },
        "drain_accounting": {
            "drain_criterion_1_p_le_2_open": "unchanged — this is P=4, outside drain scope",
            "drain_criterion_2_active_empty": "currently 1 (this exp); 0 after reviewer update",
            "net_effect": "-1 from open queue",
        },
    }

    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[{EXPERIMENT_ID}] preempt-KILL refusal written to {RESULTS_PATH.name}")
    print(f"[{EXPERIMENT_ID}] verdict=KILLED all_pass=False is_smoke=False")
    print(f"[{EXPERIMENT_ID}] blockers: F#666-pure + F#502/F#646 + predicate-not-met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
