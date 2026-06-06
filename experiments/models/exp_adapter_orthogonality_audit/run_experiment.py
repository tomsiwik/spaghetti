"""run_experiment.py — exp_adapter_orthogonality_audit (PREEMPT-KILL, F#666-pure, 2nd instance).

This experiment is preempt-killed for a KC-structural violation of guardrail
1007 (Finding #666, target-gated KILL discipline). No MLX code is written
because the pre-registered kill-criterion set consists of two proxy KCs
(K1857 pairwise cosine, K1858 effective rank) with no paired target-metric
KC. Under F#666, neither SUPPORTED nor KILLED is derivable from a proxy-only
KC set regardless of empirical outcome.

This is the SECOND F#666-pure standalone instance in the drain window (after
F#700 on `exp_g4_per_layer_cos_baseline`). Promotion trigger per analyst
watchlist: antipattern memories are filed this iteration.

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

    No MLX import or call is made. No adapters are loaded. No cosine or
    effective-rank is computed. The verdict is structural: proxy-only KC
    set cannot produce any valid verdict under F#666 regardless of
    empirical outcome.
    """
    return {
        "experiment_id": "exp_adapter_orthogonality_audit",
        "verdict": "KILLED",
        "kill_reason": "kc-structural-f666-pure-proxy-only-2nd-instance",
        "finding_reference": (
            "F#666 (target-gated KILL discipline, guardrail 1007) — "
            "2nd standalone sub-case instance (after F#700), "
            "orthogonal to F#669 family"
        ),
        "parent_experiment": None,
        "parent_status_at_claim": None,
        "all_pass": False,
        "is_smoke": False,
        "kill_criteria": [
            {
                "id": 1857,
                "text": (
                    "Pairwise cosine between any two adapter weight matrices "
                    "> 0.15 (not orthogonal)"
                ),
                "kind": "proxy",
                "result": "untested",
                "reason": (
                    "KC-structural preempt-KILL. Cosine is explicitly listed "
                    "by F#666 guardrail 1007 as a forbidden-solo proxy. No "
                    "target-metric KC is paired; per F#666 proxy-alone "
                    "verdicts are tautological regardless of pass/fail."
                ),
            },
            {
                "id": 1858,
                "text": (
                    "Effective rank of N-adapter stack < N * rank/2 "
                    "(subspace overlap > 50%)"
                ),
                "kind": "proxy",
                "result": "untested",
                "reason": (
                    "KC-structural preempt-KILL. Effective-rank / subspace-"
                    "overlap is a geometric structural statistic, not a "
                    "behavioral outcome. F#42 caveat explicitly flags that "
                    "weight-level orthogonality does NOT imply behavioral "
                    "composition success. F#571 already settled N>1 "
                    "composition as behaviorally-failing on Gemma 4 E4B. "
                    "Proxy-only KC with no target pairing; F#666 invalid."
                ),
            },
        ],
        "kc_set_gating": (
            "F#666-VIOLATING (2 proxy K1857+K1858, 0 target). Standalone "
            "F#666-pure case — no parent dependency. Orthogonal to F#698 "
            "which combined parent-unverified (F#669) + F#666 compound. "
            "2nd instance of sub-case (after F#700)."
        ),
        "secondary_structural_defects": [
            (
                "success_criteria: [] — empty; no SUPPORTED-condition "
                "declared."
            ),
            (
                "references: [] — violates guardrail 1002 (every new "
                "experiment MUST cite an arxiv paper or prior finding). "
                "Notes field references F#562 but no formal citation "
                "registered; F#42, F#571 also missing."
            ),
            (
                "platform: null — unset; MATH.md §0 discipline violated."
            ),
        ],
        "prereg_hygiene_antipattern_instance": (
            "Same 4-defect structural shape as F#700: F#666-violating KC "
            "set + empty success_criteria + empty references + null "
            "platform. 2nd instance triggers antipattern memory "
            "promotion (AP-prereg-hygiene-multi-defect)."
        ),
        "unblock_condition": (
            "KC-augmentation required (pre-registration modification): add "
            "a target-metric KC pairing K1857/K1858 to a behavioral outcome "
            "(e.g. N=3 composition retains ≥90% single-adapter task accuracy "
            "on code/math/medical domains from exp_p1_t2_single_domain_"
            "training). Add references F#42, F#562, F#571, optionally arxiv "
            "2504.10957 (Zhong et al., task arithmetic lossless conditions). "
            "Set platform=mlx. Populate success_criteria mirroring KC pass "
            "condition. Post-claim KC mutation is antipattern-u — edits must "
            "happen before re-claim. Alternative (recommended): close this "
            "pre-reg as structurally-malformed and re-register as an "
            "orthogonality-INTERVENTION experiment (does Riemannian-"
            "constrained training preserve behavioral composition at N>1?) "
            "rather than a proxy-only audit."
        ),
        "platform_skills_invoked": [
            "/mlx-dev (noted, not used — no code path)",
            "/fast-mlx (noted, not used — no code path)",
        ],
        "base_model": (
            "mlx-community/gemma-4-e4b-it-4bit (per F#627, not loaded)"
        ),
        "adapters_that_would_have_been_audited": [
            (
                "experiments/models/exp_p1_t2_single_domain_training/adapters/"
                "{code,math,medical}/adapters.safetensors"
            ),
            "plus ~10 other trained adapters on disk (not loaded)",
        ],
        "impl_follow_up_filed": False,
        "impl_follow_up_rationale": (
            "Preempt-structural KILL does NOT spawn an _impl companion "
            "(per F#687/F#698/F#699/F#700 precedent + reviewer.md §5). "
            "Unblock is pre-registration-external (requires editing DB "
            "entry to add a target KC), not implementation-external."
        ),
        "drain_subcase_taxonomy": {
            "sub_case": "F#666-pure standalone KC-structural preempt-KILL",
            "orthogonal_to": "F#669 family (parent-unverified)",
            "occurrence_index": 2,
            "prior_instance": "F#700 (exp_g4_per_layer_cos_baseline)",
            "promotion_threshold": (
                "Reached this iteration. Antipattern memory to be filed: "
                "AP-F666-pure-standalone. Also AP-prereg-hygiene-multi-"
                "defect (same 4-defect shape as F#700)."
            ),
        },
        "semantic_corroboration": (
            "F#571 already settled N>1 composition as behaviorally-failing "
            "on Gemma 4 E4B (cross-term compound under LayerNorm). Even a "
            "hypothetically runnable version of this audit would re-confirm "
            "known failure at the proxy level; research value is near-zero "
            "regardless of F#666 compliance."
        ),
        "notes": (
            "No MLX code was executed. This is a KC-structural preempt-KILL "
            "under F#666 (guardrail 1007), independent of any parent "
            "dependency. The experiment has no parent (depends_on: []) but "
            "the KC set is malformed: 2 proxy KCs (pairwise cosine + "
            "effective rank) with no target pairing. Under F#666, no "
            "outcome can produce a valid verdict. 2nd such instance in "
            "drain window (after F#700) → antipattern promotion triggered."
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
        "2nd drain-window instance"
    )


if __name__ == "__main__":
    main()
