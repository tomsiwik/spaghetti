"""run_experiment.py — exp_hedgehog_cross_axis_interference

Preempt-KILL diagnostic per F#669-family carve-out.
NOT a measurement script. Writes results.json with KILLED verdict and exits.

See MATH.md §2-§6 for the structural blocker derivation. K#1859 cannot be
measured because parent exp_hedgehog_behavior_adapter_politeness is PROVISIONAL
(no trained polite-adapter weights exist to apply on refactor prompts).
"""

import json
from pathlib import Path

RESULT = {
    "experiment_id": "exp_hedgehog_cross_axis_interference",
    "verdict": "KILLED",
    "verdict_clause": "F#669-family preempt-structural (16th reuse) + F#666-schema-defect (compound)",
    "all_pass": False,
    "is_smoke": False,
    "preempt_reason": (
        "Parent exp_hedgehog_behavior_adapter_politeness is PROVISIONAL "
        "(Phase 0 only, K#1782-K#1785 untested, no trained polite-adapter "
        "weights exist). K#1859 requires applying trained polite adapter to "
        "refactor-only prompts and measuring refactor-quality delta vs base. "
        "Measurement impossible by construction. Compound: K#1859 also single "
        "unpaired KC (F#666 schema-defect, never F#770-repaired)."
    ),
    "kill_criteria": [
        {
            "id": 1859,
            "text": (
                "Polite adapter changes refactor-quality score > 3pp on "
                "refactor-only prompts"
            ),
            "result": "fail (untested-preempt)",
            "reason": (
                "F#669-cascade: trained polite-adapter weights do not exist "
                "(parent PROVISIONAL). F#666-schema: single unpaired KC, no "
                "structural proxy counterpart for triangulation."
            ),
        },
    ],
    "findings_referenced": [
        "F#669 (governing — 16th reuse)",
        "F#666 (compound schema-defect)",
        "F#770 (~13 P<=2 schema-defect cohort, this entry NOT in cohort per F#771 audit)",
        "F#775 (1st Hedgehog-cluster F#669, post-F#770-repair sub-form)",
        "F#776 (1st schema-repair-reveals-F#669 obs)",
        "F#777 (15th F#669 reuse, 4th F#682-child, 1st post-F#770 F#682-child)",
        "F#778 (2nd schema-repair-reveals-F#669 cross-cluster obs Hedgehog->JEPA, 2/3)",
    ],
    "predicted_new_findings": [
        "F#NEW1: F#669 16th reuse, 2nd Hedgehog-cluster F#669, 1st Hedgehog-cluster pre-F#770-repair compound F#666+F#669",
        "F#NEW2: Compound F#666+F#669-pre-repair sub-form NEW sub-axis (1st instance) — distinct from F#776/F#778 schema-repair-reveals-F#669 path",
    ],
    "skill_attestation": {
        "mlx_dev_invoked": False,
        "fast_mlx_invoked": False,
        "carve_out_clause": "F#669-family (no MLX code executed, no model loaded)",
    },
    "antipattern_scan": {
        "composition_math": "N/A (no composition computed)",
        "lora_scale": "N/A (no LoRA initialized)",
        "shutil_copy_adapter": "N/A (no adapter swap)",
        "hardcoded_pass": "carved out (kill_results = fail)",
        "eval_truncation": "N/A (no eval)",
        "proxy_model_substitution": "N/A (no model loaded)",
        "kc_modification_post_hoc": "DID NOT OCCUR (K#1859 byte-for-byte identical to 2026-04-23 DB)",
    },
    "scope_preservation_shortcuts_rejected": [
        "Surrogate adapter (random-init): measures noise, not axis-independence",
        "Skip K#1859 + pass on null: F#666 hardcoded-pass antipattern",
        "Reduce K#1859 threshold from 3pp: KC modification disallowed",
        "Substitute parent Phase 0 neutral-prompt results: different axis entirely",
        "Wait for parent _impl SUPPORTED: correct path, exceeds 90-min researcher cap",
    ],
    "drain_diagnostic": {
        "consecutive_researcher_preempt_kills": 3,
        "iters": "rank_ablation~47 -> jepa_scale_sweep~49 -> cross_axis_interference~52",
        "doom_loop_check": "substantively distinct (different cluster/mechanism/finding-index per iter)",
        "in_cap_progress_path": "preempt-KILL of cascade children until macro _impl budgets unlocked",
    },
}


def main():
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(RESULT, indent=2) + "\n")
    print(f"WROTE {out} ({out.stat().st_size} bytes)")
    print(f"VERDICT: {RESULT['verdict']} ({RESULT['verdict_clause']})")


if __name__ == "__main__":
    main()
