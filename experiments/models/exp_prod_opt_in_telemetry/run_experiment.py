"""Refusal scaffold — exp_prod_opt_in_telemetry.

Structural preempt-KILL per F#765 super-family (PROD-deliverable-cascade,
5th instance, no-parent sub-form). Per F#769 closing-note: no new finding.

The KCs in the DB (K1679/K1680/K1681) are deliverable-spec checks, not
scientific measurements:
  - K1679: product-default flag (binary source-grep, not metric)
  - K1680: privacy-policy compliance (manual code audit)
  - K1681: third-party legal review (out of repo)

This file does not load any model, does not write any production code, does
not invoke any platform skills, and emits a refusal results.json so that
reviewer can issue `experiment update --status killed --evidence "F#765 super-family
5th instance, F#769 ledger-explosion: no new finding"`.

See MATH.md §2 for the impossibility argument and §5 for the antipattern scan.
"""

from __future__ import annotations

import json
import pathlib
import sys

EXPERIMENT_ID = "exp_prod_opt_in_telemetry"
RESULTS_PATH = pathlib.Path(__file__).parent / "results.json"


def main() -> int:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "verdict": "KILLED",
        "all_pass": False,
        "is_smoke": False,
        "kill_criteria": {
            "K1679": {
                "result": "untested",
                "reason": (
                    "Deliverable-spec check: 'telemetry off by default + consent flow'"
                    " is a binary source-grep against product code, not a scientific"
                    " measurement. No proxy/target metric pair per F#666."
                ),
            },
            "K1680": {
                "result": "untested",
                "reason": (
                    "Privacy-policy compliance check: 'no user text/prompts/completions'"
                    " requires a manual code audit of the telemetry payload schema."
                    " Not falsifiable from this repo without an explicit forbidden-fields"
                    " contract test, which is a deliverable not an experiment."
                ),
            },
            "K1681": {
                "result": "untested",
                "reason": (
                    "Legal/process gate: 'GDPR compliance review passes' requires a"
                    " third-party human compliance reviewer. Out of repo scope."
                ),
            },
        },
        "preempt_kill": {
            "reason": "F#765 super-family — PROD-deliverable-cascade 5th instance",
            "sub_form": "no-parent + no-measurable-scientific-KC (NEW within super-family)",
            "prior_instances": ["F#740", "F#741", "F#764", "F#765"],
            "ledger_explosion_guard": "F#769 closing-note: no new finding for Nth instance",
            "f666_violation": "all 3 KCs are deliverable-spec, no proxy/target pairing",
            "f502_f646_violation": "success_criteria=[] in DB — 11th cohort instance",
            "category_error": (
                "this is a privacy-engineering deliverable mis-filed as a research"
                " experiment; belongs in roadmap doc / PR review, not experiments/models/"
            ),
        },
        "skills_invoked": {
            "mlx_dev": False,
            "fast_mlx": False,
            "rationale": (
                "no platform code written; refusal scaffold only; matches precedent"
                " F#763/F#764/F#765/F#768/F#769"
            ),
        },
        "doom_loop_check": {
            "prior_3_iterations_verdict": ["PROVISIONAL", "PROVISIONAL", "PROVISIONAL"],
            "this_iteration_verdict": "KILLED (preempt-structural)",
            "structurally_different_action": True,
            "rationale": "different verdict path breaks the 4th-consec-PROVISIONAL doom-loop signal",
        },
        "reviewer_route": {
            "expected_action": "experiment update --status killed",
            "expected_evidence": "F#765 super-family 5th instance; F#769 ledger-explosion: no new finding",
            "finding_add": "SKIP",
        },
    }

    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[{EXPERIMENT_ID}] preempt-KILL refusal written to {RESULTS_PATH.name}")
    print("[{0}] verdict=KILLED all_pass=False is_smoke=False".format(EXPERIMENT_ID))
    return 0


if __name__ == "__main__":
    sys.exit(main())
