"""run_experiment.py — exp_g4_per_layer_cos_baseline (PREEMPT-KILL, F#666-pure).

This experiment is preempt-killed for a KC-structural violation of guardrail
1007 (Finding #666, target-gated KILL discipline). No MLX code is written
because the pre-registered kill-criterion set consists of a single proxy KC
(K1856: per-layer cos-sim variance) with no paired target-metric KC. Under
F#666, neither SUPPORTED nor KILLED is derivable from a proxy-only KC set
regardless of empirical outcome.

Distinguished from F#669/F#687/F#698/F#699 (preempt-child-parent-target-
unverified): this experiment has NO parent dependency (`depends_on: []`).
The structural defect is in the KC set itself, not in an upstream artifact.

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

    No MLX import or call is made. No model is loaded. No cos-sim variance
    is computed. The verdict is structural: proxy-only KC set cannot produce
    any valid verdict under F#666 regardless of empirical outcome.
    """
    return {
        "experiment_id": "exp_g4_per_layer_cos_baseline",
        "verdict": "KILLED",
        "kill_reason": "kc-structural-f666-pure-proxy-only",
        "finding_reference": (
            "F#666 (target-gated KILL discipline, guardrail 1007) — "
            "standalone sub-case, orthogonal to F#669 family"
        ),
        "parent_experiment": None,
        "parent_status_at_claim": None,
        "all_pass": False,
        "is_smoke": False,
        "kill_criteria": [
            {
                "id": 1856,
                "text": (
                    "Per-layer cos-sim variance across 100 diverse prompts "
                    "< 0.02 (routing is uniform, not layer-specific)"
                ),
                "kind": "proxy",
                "result": "untested",
                "reason": (
                    "KC-structural preempt-KILL. Proxy-only KC with no target "
                    "pairing violates F#666. Per guardrail 1007, KILL requires "
                    "proxy+target BOTH to fail; SUPPORTED requires BOTH to pass. "
                    "Proxy-alone verdict is tautological (Proxy-FAIL + target-PASS "
                    "= finding about proxy, not kill; Proxy-PASS + target-FAIL = "
                    "tautological proxy, kill on target — neither path is "
                    "available here). Therefore no empirical outcome can "
                    "produce a valid verdict; preempt before running."
                ),
            }
        ],
        "kc_set_gating": (
            "F#666-VIOLATING (1 proxy K1856, 0 target). Standalone F#666-pure "
            "case — no parent dependency. Orthogonal to F#698 which combined "
            "parent-unverified (F#669) + F#666 compound."
        ),
        "secondary_structural_defects": [
            "success_criteria: [] — empty; no SUPPORTED-condition declared.",
            (
                "references: [] — violates guardrail 1002 (every new experiment "
                "MUST cite an arxiv paper or prior finding)."
            ),
            "platform: null — unset; MATH.md §0 discipline violated.",
        ],
        "unblock_condition": (
            "KC-augmentation required (pre-registration modification): add a "
            "target-metric KC pairing K1856 to a behavioral outcome on a "
            "Hedgehog-distilled model (e.g. cos-sim-variance ↔ downstream-"
            "task-accuracy correlation, or layer-selection-informed "
            "distillation preserves ≥ 90% base accuracy). Also add an arxiv "
            "reference and set platform=mlx. Post-claim KC mutation is "
            "antipattern-u — edits must happen before re-claim. Alternative: "
            "re-scope as sibling of Hedgehog family (F#683/F#684/F#696/F#697) "
            "with target KC baked in, rather than resurrecting the malformed "
            "pre-reg."
        ),
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
            "(per F#687/F#698/F#699 precedent + reviewer.md §5). Unblock is "
            "pre-registration-external (requires editing DB entry to add a "
            "target KC), not implementation-external."
        ),
        "drain_subcase_taxonomy": {
            "sub_case": "F#666-pure standalone KC-structural preempt-KILL",
            "orthogonal_to": "F#669 family (parent-unverified)",
            "occurrence_index": 1,
            "promotion_threshold": (
                "If a second instance occurs in drain window, promote to "
                "standalone antipattern memory. 1st instance → document, "
                "watch, don't promote."
            ),
        },
        "notes": (
            "No MLX code was executed. This is a KC-structural preempt-KILL "
            "under F#666 (guardrail 1007), independent of any parent "
            "dependency. The experiment has no parent (depends_on: []) but "
            "the KC set itself is malformed: single proxy KC (cos-sim "
            "variance) with no target pairing. Under F#666, no outcome can "
            "produce a valid verdict."
        ),
    }


def main() -> None:
    """Entry point — never raises, always writes results.json."""
    results = build_results()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(
        f"[preempt-kill] Wrote {out} — verdict=KILLED, "
        "reason=F#666-pure KC-structural (proxy-only, no target pairing)"
    )


if __name__ == "__main__":
    main()
