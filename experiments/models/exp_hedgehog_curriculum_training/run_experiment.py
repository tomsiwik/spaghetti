"""exp_hedgehog_curriculum_training -- preempt-structural KILL stub.

No measurement is performed. The verdict is pre-registered KILLED per MATH.md:
F#666-pure standalone, 6th Hedgehog-ablation sub-type (1st curriculum /
training-procedure-ablation; cousin of F#722 hyperparameter-ablation, F#723
data-augmentation-ablation). The pre-registered KC set K = {K1933, K1934} is
proxy-only with no paired target KC, which under F#666 / guardrail 1007 admits
no compliant verdict regardless of measurement outcome:
  - both PASS  -> tautological KILL (curriculum cos-sim worse, but no behavioral evidence)
  - any FAIL   -> "finding about proxy, not kill" per F#666 explicit rule

This script intentionally writes results.json with verdict=KILLED and exits
without loading any model, tokenizer, dataset, teacher, student, or trainer.
The re-registration path is described in MATH.md section 5.
"""

import json
from pathlib import Path

RESULTS = {
    "experiment_id": "exp_hedgehog_curriculum_training",
    "verdict": "KILLED",
    "preempt_reason": "F666_PURE_STANDALONE_PROXY_ONLY_KC_SET_HEDGEHOG_ABLATION_6TH_SUBTYPE",
    "sub_flavor_primary": "hedgehog_ablation_curriculum_training_procedure",
    "sub_flavor_primary_index_in_hedgehog_ablation_super_family": 6,
    "sub_flavor_secondary": "cos_sim_bucket_convergence_speed",
    "sub_flavor_secondary_index_in_cos_sim_bucket": 2,
    "all_pass": False,
    "is_smoke": False,
    "kill_criteria": [
        {
            "id": 1933,
            "text": "Curriculum training produces adapter > 3pp worse than random-order training",
            "kind": "proxy",
            "result": "untested",
            "reason": (
                "Relative cos-sim/PPL adapter quality delta. Hedgehog "
                "framework default metric is cos-sim against teacher per "
                "layer (Moudgil sec 3.1). Even if 'worse' is interpreted as "
                "PPL, guardrail 1006 declares PPL r~0.08 to task quality "
                "in this codebase -- PPL is itself a proxy. Delta-of-proxies "
                "is still a proxy (F#754 sec 1.1 invariant). No coupling to "
                "behavioral outcome (LLM-judge politeness score per F#683 "
                "K1783 not measured). 1st curriculum-ablation instance in "
                "Hedgehog-ablation 6th sub-type; closest sibling F#722 "
                "(hyperparameter-ablation, teacher-temperature sweep, both "
                "KCs proxy, killed)."
            ),
        },
        {
            "id": 1934,
            "text": "Curriculum training cos-sim convergence < random-order (worse)",
            "kind": "proxy",
            "result": "untested",
            "reason": (
                "Training-curve cos-sim convergence speed. Direct cos-sim "
                "measurement during training -- explicitly cos-sim per KC "
                "text. F#720 precedent (1st cos-sim-bucket instance, MSE "
                "loss-variant) preempt-killed sole cos-sim KC; this is 2nd "
                "cos-sim-bucket instance in convergence-speed form (F#720 "
                "was final-value form). Speed-of-cos-sim inherits "
                "cos-sim-as-proxy. F#702 precedent (latency + bitwise-exact "
                "equivalence) shows runnable training-curve form -- pair "
                "with behavioral invariant; K1934 has no such pair."
            ),
        },
    ],
    "kc_set_gating": (
        "F#666-VIOLATING (2 proxy K1933+K1934, 0 target). Standalone "
        "F#666-pure case -- no parent dependency (depends_on=[]). "
        "6th Hedgehog-ablation super-family sub-type instance "
        "(curriculum / training-procedure-ablation; previously "
        "axis-extension, loss-variant F#719/F#720, layer-selection F#721, "
        "hyperparameter F#722, data-augmentation F#723). ~25th drain-window "
        "F#666-pure-standalone instance (after F#700, F#701, F#703, F#705, "
        "F#706, F#707, F#708, F#710, F#711, F#714, F#715, F#716, F#720, "
        "F#722, F#728, F#729, F#730, F#731, F#732, F#734, F#735, F#736, "
        "F#753, F#754)."
    ),
    "secondary_structural_defects": [
        "success_criteria: NONE -- DB explicitly flags missing.",
        "platform: null -- DB hygiene defect (DB explicitly flags missing).",
        "references: [] -- violates guardrail 1002 (must cite arxiv paper or prior finding); none provided despite extensive curriculum-learning prior art (Bengio 2009 arxiv:0903.0738; Hacohen-Weinshall 2019 arxiv:1904.03626; Wu et al. 2021 arxiv:2010.13166 curriculum-for-distillation).",
        "experiment_dir: null until this iteration created the dir.",
    ],
    "hygiene_defects_count": 4,
    "hygiene_multi_defect_threshold_crossed": True,
    "platform_skills_invoked": [
        "/mlx-dev (noted, not used -- no code path)",
        "/fast-mlx (noted, not used -- no code path)",
    ],
    "base_model": "mlx-community/gemma-4-e4b-it-4bit (per F#627, not loaded)",
    "implicit_conceptual_parent": (
        "exp_hedgehog_behavior_adapter_politeness_impl (P=1, status=open, "
        "never executed). Even with target-pair KCs, this experiment would "
        "be F#669-style child-on-unverified-parent: comparing two curricula "
        "of an unverified base method has no anchor. Curriculum-vs-random "
        "is meaningful only if random-order itself produces a working "
        "adapter."
    ),
    "prior_partial_coverage": {
        "K1933_curriculum_vs_random_quality_delta": [
            {"finding": 722, "claim": "Hedgehog teacher-temperature sweep (4th Hedgehog-ablation sub-type, hyperparameter-ablation): both KCs proxy, killed -- direct sibling structural shape"},
            {"finding": 720, "claim": "Hedgehog MSE loss-variant ablation (1st cos-sim-bucket, intra-loss-function-delta): K1872 cos-sim only, killed"},
            {"finding": 723, "claim": "Hedgehog data-augmentation-ablation (5th Hedgehog-ablation sub-type) WITH target-pair K1877+K1878 -> PROVISIONAL not killed; demonstrates runnable design pattern"},
            {"arxiv": "2010.13166", "claim": "Wu et al. 'Curriculum Learning for Knowledge Distillation' includes student-task accuracy (target) alongside KL/cos-sim convergence (proxy) -- runnable template"},
        ],
        "K1934_cos_sim_convergence_speed": [
            {"finding": 720, "claim": "K1872 sole cos-sim KC preempt-killed (1st cos-sim-bucket, final-value form). K1934 is 2nd cos-sim-bucket instance, convergence-speed form -- inherits proxy classification"},
            {"finding": 702, "claim": "Latency + bitwise-exact equivalence is the canonical runnable training-curve metric pair; K1934 has no such behavioral pair"},
        ],
        "note": (
            "Curriculum-learning-for-distillation has substantial published "
            "prior art with target-pair runnable designs (Wu 2021); this "
            "experiment's failure is structural (KC design), not topical. "
            "F#722 (sibling 4th Hedgehog-ablation sub-type) and F#723 "
            "(5th sub-type) are direct precedents; F#723's K1877+K1878 "
            "is the design pattern the curriculum experiment must adopt."
        ),
    },
    "impl_follow_up_filed": False,
    "impl_follow_up_rationale": (
        "Preempt-structural KILL does NOT spawn an _impl companion (per "
        "F#687/F#698/F#699/F#700/F#701/F#703/F#705/F#710/F#714/F#722/F#753/"
        "F#754 precedent + reviewer.md section 5). Unblock is "
        "pre-registration-external (requires editing DB entry to add "
        "target-pair KC + references + platform), not implementation-external."
    ),
    "drain_subcase_taxonomy": {
        "primary_sub_case": "F#666-pure standalone, Hedgehog-ablation super-family 6th sub-type (curriculum/training-procedure-ablation)",
        "primary_occurrence_index_in_hedgehog_ablation": 6,
        "secondary_sub_case": "cos-sim-bucket 2nd instance (convergence-speed form)",
        "secondary_occurrence_index_in_cos_sim_bucket": 2,
        "orthogonal_to": "F#669 family (parent-unverified) [though implicit conceptual parent F#683 is unverified]",
        "depends_on": [],
        "approx_overall_index": 25,
        "prior_hedgehog_ablation_subtypes": [
            "axis-extension (F#683, F#684, F#696, F#697 -- behavior-axis instances)",
            "loss-variant-ablation (F#719 K1870+K1871 target-pair runnable; F#720 K1872 cos-sim-only killed)",
            "layer-selection-ablation (F#721 triple-fire preempt-KILL)",
            "hyperparameter-ablation (F#722 teacher-temperature sweep, triple-fire preempt-KILL)",
            "data-augmentation-ablation (F#723 K1877 target + K1878 proxy runnable -> PROVISIONAL)",
        ],
        "this_subtype_position": "6th sub-type, 1st curriculum/training-procedure-ablation instance",
        "closest_structural_sibling": "F#722 (hyperparameter-ablation, both KCs proxy, killed)",
        "closest_runnable_template": "F#723 (data-augmentation-ablation, K1877 target + K1878 proxy)",
    },
    "notes": (
        "Preempt-structural KILL -- no model loaded, no dataset opened, "
        "no tokenizer invoked, no teacher forward pass, no student forward "
        "pass, no per-layer cos-sim computed, no curriculum schedule "
        "constructed, no random-order baseline trained. "
        "See MATH.md for theorem + 4-cell verdict truth table + unblock "
        "condition (re-register exp_hedgehog_curriculum_training_behavioral "
        "with target-pair KC, references, and post-F#683-supported)."
    ),
}


def main() -> int:
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(RESULTS, indent=2))
    print(f"Wrote {out}")
    print(f"Verdict: {RESULTS['verdict']} ({RESULTS['preempt_reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
