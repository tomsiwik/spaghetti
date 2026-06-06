"""run_experiment.py — exp_hedgehog_triple_composition_3domain

Preempt-KILL diagnostic per F#669-family carve-out.
NOT a measurement script. Writes results.json with KILLED verdict and exits.

See MATH.md §2-§7 for the structural blocker derivation. K#1883 and K#1884
cannot be measured because ALL THREE parents (python_domain + sql_domain + js)
are PROVISIONAL with all KCs untested AND no adapters/ subdir exists in any
dep — none of the trained adapters exist, so the triple composition is
unconstructable. NEW: 3-parent F#669 cardinality (highest dep-cardinality
observed in F#669 family).
"""

import json
from pathlib import Path

RESULT = {
    "experiment_id": "exp_hedgehog_triple_composition_3domain",
    "verdict": "KILLED",
    "verdict_clause": (
        "F#669-family preempt-structural (18th reuse; 3-parent cardinality "
        "1st observation, highest dep-cardinality ever in F#669 family) + "
        "F#666-pure schema-defect (both KCs proxy-only, compound)"
    ),
    "all_pass": False,
    "is_smoke": False,
    "preempt_reason": (
        "All three parents PROVISIONAL with all KCs untested and no trained "
        "adapter weights on disk: exp_hedgehog_adapter_python_domain "
        "(K#1844-K#1845 untested, no adapters/), exp_hedgehog_adapter_sql_domain "
        "(K#1868-K#1869 untested, no adapters/), exp_hedgehog_domain_adapter_js "
        "(K#1790-K#1793 untested, no adapters/). K#1883 requires applying "
        "isolated AND triple-composed py+sql+js adapters to per-domain "
        "prompts; none of the three trained adapters exist. K#1884 requires "
        "per-layer cos-sim of triple vs 3-prompt teacher; unconstructable. "
        "Compound: K#1883 + K#1884 both proxy-only KCs (no target-triple, "
        "F#666-pure schema-defect)."
    ),
    "kill_criteria": [
        {
            "id": 1883,
            "text": (
                "Triple domain composition drops any single domain > 5pp vs "
                "isolated"
            ),
            "result": "fail (untested-preempt)",
            "reason": (
                "F#669-cascade 3-parent: trained python-adapter, "
                "sql-adapter, AND js-adapter weights all do not exist. "
                "Triple composition is unconstructable. ls confirms no "
                "adapters/ subdir in any of the three dep directories."
            ),
        },
        {
            "id": 1884,
            "text": (
                "Per-layer cos of triple composition vs 3-prompt "
                "concatenated teacher < 0.60"
            ),
            "result": "fail (untested-preempt)",
            "reason": (
                "F#669-cascade 3-parent: same as K#1883 — triple adapter "
                "does not exist as trained artifact (none of 3 components "
                "are trained)."
            ),
        },
    ],
    "findings_referenced": [
        "F#669 (governing — 18th reuse, 3-parent cardinality 1st obs)",
        "F#666 (compound schema-defect; F#666-pure both-proxy variant)",
        "F#770 (~13 P<=2 schema-defect cohort; this entry NOT in cohort)",
        "F#771 (audit-correction reducing F#770 cohort)",
        "F#775 (1st Hedgehog-cluster F#669, post-F#770-repair sub-form)",
        "F#777 (15th F#669, 4th F#682-child)",
        "F#779 (16th F#669, 1st Hedgehog pre-F#770-repair compound)",
        "F#780 (compound F#666+F#669 pre-F#770-repair sub-axis 1st obs)",
        "F#781 (17th F#669, 2-parent F#669 cardinality 1st obs)",
        "F#782 (F#780 sub-axis 2nd-instance same-cluster Hedgehog)",
        "F#683 (Hedgehog parent finding)",
        "F#752 (composition residual tau~0.48; secondary antipattern)",
    ],
    "predicted_new_findings": [
        (
            "F#NEW1: F#669 18th reuse; 3-parent F#669 cardinality 1st "
            "observation (highest dep-cardinality ever in F#669 family); "
            "4th Hedgehog-cluster F#669; 3rd Hedgehog-cluster "
            "pre-F#770-repair compound F#666+F#669"
        ),
        (
            "F#NEW2: F#780 sub-axis 3rd-instance same-cluster "
            "(Hedgehog->Hedgehog->Hedgehog), advancing 2/3 -> 3/3 "
            "same-cluster canonicalization saturation. Cross-cluster "
            "canonicalization (3-cluster diversity) remains pending — "
            "same-cluster 3-of-3 is full saturation within Hedgehog cluster."
        ),
    ],
    "skill_attestation": {
        "mlx_dev_invoked": False,
        "fast_mlx_invoked": False,
        "carve_out_clause": "F#669-family (no MLX code executed, no model loaded, no composition computed)",
    },
    "antipattern_scan": {
        "composition_math": "N/A (no composition computed); §4 notes F#752 tau~0.48 ceiling secondary concern, exacerbated by N=3 vs F#752's N=2 measurement",
        "lora_scale": "N/A (no LoRA initialized)",
        "shutil_copy_adapter": "N/A (no adapter swap)",
        "hardcoded_pass": "carved out (kill_results = fail)",
        "eval_truncation": "N/A (no eval)",
        "proxy_model_substitution": "N/A (no model loaded)",
        "kc_modification_post_hoc": "DID NOT OCCUR (K#1883 + K#1884 byte-for-byte identical to 2026-04-23 DB)",
    },
    "scope_preservation_shortcuts_rejected": [
        "Surrogate adapters (random-init all three): measures noise, not triple-composition",
        "Identity-adapter substitution: measures base model, not triple",
        "Skip K#1883+K#1884, hardcode pass:True: F#666 hardcoded-pass antipattern",
        "Reduce 5pp / 0.60 thresholds: KC modification disallowed (guardrail 1010)",
        "Train just one of the 3 deps as N=1 surrogate: changes axis (K#1883 measures triple, not single)",
        "Wait for all 3 parents _impl SUPPORTED: correct path; no _impl exists for py or sql; only js has _impl filed (still blocked-by parent)",
    ],
    "drain_diagnostic": {
        "preempt_kill_pattern_state": "post-HALT-override drain progress",
        "iters": (
            "rank_ablation~47 -> jepa_scale_sweep~49 -> cross_axis~52 -> "
            "pair_composition~55 -> [HALT-override drain progress: "
            "politeness_impl~58/F#783, refactor_impl~61/F#784, "
            "rdt_loop_kv_cache_impl~64/F#785, formality_impl~67/F#786, "
            "rdt_loop_kv_cache_full~70-91/F#787-F#788, conciseness_impl~92/F#789, "
            "conciseness_full~94/F#790] -> triple_composition~97 (this)"
        ),
        "doom_loop_check": (
            "1st preempt-KILL since 7 consecutive HALT-override smoke iters "
            "broke the F#669-cascade pattern. Substantively distinct "
            "sub-mechanism: 3-parent F#669 cardinality NEVER previously "
            "observed (highest dep-cardinality ever in F#669 family). "
            "F#780 sub-axis 3rd same-cluster instance reaches "
            "canonicalization saturation."
        ),
        "in_cap_progress_path_remaining": (
            "0 (after this preempted). All remaining P<=2 entries are "
            "P=1 macro _impl with multi-hour budgets."
        ),
        "halt_escalation_addendum_required": False,
        "drain_post_iter_state": (
            "P<=2 open=5 (memento_replication, class_composition_full_impl, "
            "politeness_full, refactor_full, formality_full). All P=1 macro "
            "_impl. Drain advances via macro-orchestration, not "
            "researcher-cap claims."
        ),
    },
}


def main():
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(RESULT, indent=2) + "\n")
    print(f"WROTE {out} ({out.stat().st_size} bytes)")
    print(f"VERDICT: {RESULT['verdict']} ({RESULT['verdict_clause']})")


if __name__ == "__main__":
    main()
