"""run_experiment.py — exp_hedgehog_pair_composition_polite_refactor

Preempt-KILL diagnostic per F#669-family carve-out.
NOT a measurement script. Writes results.json with KILLED verdict and exits.

See MATH.md §2-§7 for the structural blocker derivation. K#1846 and K#1847
cannot be measured because BOTH parents (politeness + refactor) are PROVISIONAL
with all 4 KCs each untested — neither trained adapter exists, so the pair
composition is unconstructable.
"""

import json
from pathlib import Path

RESULT = {
    "experiment_id": "exp_hedgehog_pair_composition_polite_refactor",
    "verdict": "KILLED",
    "verdict_clause": (
        "F#669-family preempt-structural (17th reuse; 2-parent cardinality "
        "1st observation) + F#666-pure schema-defect (both KCs proxy-only, "
        "compound)"
    ),
    "all_pass": False,
    "is_smoke": False,
    "preempt_reason": (
        "Both parents PROVISIONAL with all 4 KCs each untested: "
        "exp_hedgehog_behavior_adapter_politeness (Phase 0 only, K#1782-K#1785 "
        "untested) AND exp_hedgehog_procedural_adapter_refactor (design only, "
        "K#1786-K#1789 untested). K#1846 requires applying isolated and "
        "pair-composed polite+refactor adapters to behavior+code prompts; "
        "neither trained adapter exists. K#1847 requires per-layer cos-sim of "
        "pair vs 2-prompt teacher; unconstructable. Compound: K#1846 + K#1847 "
        "both proxy-only KCs (no target-pair, F#666-pure schema-defect, never "
        "F#770-repaired)."
    ),
    "kill_criteria": [
        {
            "id": 1846,
            "text": (
                "Pair-composed polite+refactor drops either axis > 5pp vs "
                "isolated"
            ),
            "result": "fail (untested-preempt)",
            "reason": (
                "F#669-cascade 2-parent: trained polite-adapter AND trained "
                "refactor-adapter weights both do not exist. Pair composition "
                "is unconstructable."
            ),
        },
        {
            "id": 1847,
            "text": (
                "Per-layer cos of pair composition vs 2-prompt concatenated "
                "teacher < 0.65"
            ),
            "result": "fail (untested-preempt)",
            "reason": (
                "F#669-cascade 2-parent: same as K#1846 — pair adapter does "
                "not exist as trained artifact."
            ),
        },
    ],
    "findings_referenced": [
        "F#669 (governing — 17th reuse, 2-parent cardinality 1st obs)",
        "F#666 (compound schema-defect; F#666-pure both-proxy variant)",
        "F#770 (~13 P<=2 schema-defect cohort, this entry NOT in cohort per F#771 audit)",
        "F#771 (audit-correction reducing F#770 cohort)",
        "F#775 (1st Hedgehog-cluster F#669, post-F#770-repair sub-form)",
        "F#776 (1st schema-repair-reveals-F#669 obs)",
        "F#777 (15th F#669, 4th F#682-child)",
        "F#778 (2nd schema-repair-reveals-F#669 cross-cluster Hedgehog->JEPA, 2/3)",
        "F#779 (16th F#669, 1st Hedgehog pre-F#770-repair compound)",
        "F#780 (compound F#666+F#669 pre-F#770-repair sub-axis 1st obs)",
        "F#683 (Hedgehog parent finding)",
        "F#752 (composition residual tau~0.48, secondary antipattern concern)",
    ],
    "predicted_new_findings": [
        (
            "F#NEW1: F#669 17th reuse; 2-parent F#669 cardinality 1st "
            "observation; 3rd Hedgehog-cluster F#669; 2nd Hedgehog-cluster "
            "pre-F#770-repair compound F#666+F#669"
        ),
        (
            "F#NEW2: F#780 sub-axis 2nd-instance same-cluster (Hedgehog->Hedgehog), "
            "advancing 1/3 -> 2/3 toward canonicalization. NOT cross-cluster "
            "(same-cluster only). Distinct from F#776/F#778 schema-repair-"
            "reveals-F#669 path."
        ),
    ],
    "skill_attestation": {
        "mlx_dev_invoked": False,
        "fast_mlx_invoked": False,
        "carve_out_clause": "F#669-family (no MLX code executed, no model loaded, no composition computed)",
    },
    "antipattern_scan": {
        "composition_math": "N/A (no composition computed); §4 notes F#752 tau~0.48 ceiling secondary concern if parents reached SUPPORTED",
        "lora_scale": "N/A (no LoRA initialized)",
        "shutil_copy_adapter": "N/A (no adapter swap)",
        "hardcoded_pass": "carved out (kill_results = fail)",
        "eval_truncation": "N/A (no eval)",
        "proxy_model_substitution": "N/A (no model loaded)",
        "kc_modification_post_hoc": "DID NOT OCCUR (K#1846 + K#1847 byte-for-byte identical to 2026-04-23 DB)",
    },
    "scope_preservation_shortcuts_rejected": [
        "Surrogate adapters (random-init both axes): measures noise, not pair-composition",
        "Identity-adapter substitution: measures base model, not pair",
        "Skip K#1846+K#1847, hardcode pass:True: F#666 hardcoded-pass antipattern",
        "Reduce 5pp / 0.65 thresholds: KC modification disallowed (guardrail 1010)",
        "Use parent Phase 0 prompts as proxy: different axis entirely (no pair composition)",
        "Wait for parents _impl SUPPORTED: correct path, both exceed 90-min researcher cap",
    ],
    "drain_diagnostic": {
        "consecutive_researcher_preempt_kills": 4,
        "iters": "rank_ablation~47 -> jepa_scale_sweep~49 -> cross_axis~52 -> pair_composition~55",
        "doom_loop_check": (
            "4th consecutive preempt: substantively distinct sub-mechanism "
            "(2-parent F#669 cardinality NEVER previously observed; F#780 "
            "sub-axis 2nd-instance). BUT pattern itself has canonicalized "
            "per mem-pattern-triple-fire 3-instance threshold. HALT_ESCALATION "
            "addendum requested at analyst pass."
        ),
        "in_cap_progress_path_remaining": (
            "1 (triple_composition P=2 micro). After that preempted: 0 in-cap "
            "P<=2 paths. Orchestrator must promote politeness_impl + refactor_impl "
            "macros to release drain."
        ),
        "halt_escalation_addendum_required": True,
    },
}


def main():
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(RESULT, indent=2) + "\n")
    print(f"WROTE {out} ({out.stat().st_size} bytes)")
    print(f"VERDICT: {RESULT['verdict']} ({RESULT['verdict_clause']})")


if __name__ == "__main__":
    main()
