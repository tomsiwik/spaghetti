"""exp_routing_latency_benchmark_all -- preempt-structural KILL stub.

No measurement is performed. The verdict is pre-registered KILLED per MATH.md:
F#666-pure standalone, multi-bucket sub-flavor (3rd routing-accuracy + 2nd
infrastructure-benchmark). The pre-registered KC set K = {K1929, K1930} is
proxy-only with no paired target KC, which under F#666 / guardrail 1007
admits no compliant verdict regardless of measurement outcome:
  - both PASS  -> tautological support (F#666 canonical 40.2% routing-acc + 0.0% target gap)
  - any FAIL   -> "finding about proxy, not kill" per F#666 explicit rule

This script intentionally writes results.json with verdict=KILLED and exits
without loading any model, tokenizer, dataset, or routing classifier. The
re-registration path is described in MATH.md section 5.
"""

import json
from pathlib import Path

RESULTS = {
    "experiment_id": "exp_routing_latency_benchmark_all",
    "verdict": "KILLED",
    "preempt_reason": "F666_PURE_STANDALONE_PROXY_ONLY_KC_SET_MULTI_BUCKET",
    "sub_flavor_primary": "routing_accuracy",
    "sub_flavor_primary_instance": 3,
    "sub_flavor_secondary": "infrastructure_benchmark_latency_only",
    "sub_flavor_secondary_instance": 2,
    "all_pass": False,
    "is_smoke": False,
    "kill_criteria": [
        {
            "id": 1929,
            "text": "Any routing method > 10ms per query (too slow for real-time)",
            "kind": "proxy",
            "result": "untested",
            "reason": (
                "Pure latency threshold with no behavioral anchor. Random or "
                "constant routing trivially passes (fast but useless). Per "
                "F#666 / guardrail 1007 a latency proxy without paired "
                "behavioral target (cf. F#702 latency+bitwise-equivalence "
                "pair) cannot support a compliant verdict."
            ),
        },
        {
            "id": 1930,
            "text": "Best routing method accuracy < 80% at N=25",
            "kind": "proxy",
            "result": "untested",
            "reason": (
                "Routing match rate (= 'classification accuracy' per F#666 "
                "guardrail 1007 explicit enumeration). 3rd routing-acc "
                "sub-flavor instance (F#703 1st, F#710 2nd, this 3rd). F#666 "
                "canonical: 40.2% acc + 0.0% target gap demonstrates "
                "match-rate decoupled from utility via semantic-cluster "
                "routing."
            ),
        },
    ],
    "kc_set_gating": (
        "F#666-VIOLATING (2 proxy K1929+K1930, 0 target). Standalone "
        "F#666-pure case -- no parent dependency (depends_on=[]). "
        "Multi-bucket fire: routing-accuracy 3rd sub-flavor + "
        "infrastructure-benchmark 2nd sub-flavor. ~23rd drain-window "
        "F#666-pure-standalone instance (after F#700, F#701, F#703, F#705, "
        "F#706, F#707, F#708, F#710, F#711, F#714, F#722, F#728, F#729, "
        "F#730, F#731, F#732, F#734, ...)."
    ),
    "secondary_structural_defects": [
        "success_criteria: [] -- empty; no SUPPORTED-condition declared.",
        "platform: null -- DB hygiene defect (guardrail).",
        "references: [] -- violates guardrail 1002 (must cite arxiv paper or prior finding); none provided despite extensive prior art (F#108, F#144, F#145, F#147, F#171, F#251, F#257, F#431, F#666).",
    ],
    "hygiene_defects_count": 3,
    "hygiene_multi_defect_threshold_crossed": True,
    "platform_skills_invoked": [
        "/mlx-dev (noted, not used -- no code path)",
        "/fast-mlx (noted, not used -- no code path)",
    ],
    "base_model": "mlx-community/gemma-4-e4b-it-4bit (per F#627, not loaded)",
    "prior_partial_coverage": {
        "K1929_latency": [
            {"finding": 108, "claim": "Hash ring routing latency negligible (sub-microsecond)"},
            {"finding": 144, "claim": "All routing strategies <5us latency at N=100"},
            {"finding": 145, "claim": "Routing latency solved at N=1000 (6 strategies)"},
        ],
        "K1930_accuracy": [
            {"finding": 251, "claim": "TF-IDF logistic routing 96.6% at N=5"},
            {"finding": 431, "claim": "TF-IDF routing scales to N=25 weighted acc 86.1%"},
            {"finding": 171, "claim": "Routing mechanisms survey: 5 recommendations for N>25"},
            {"finding": 147, "claim": "xxHash32 best hash for SOLE routing"},
        ],
        "note": (
            "Both proxy KCs already have prior-art answers in pre-F#666 "
            "regime. Even setting aside F#666 violation, the experiment "
            "is largely redundant. Behavioral re-frame is needed for new value."
        ),
    },
    "impl_follow_up_filed": False,
    "impl_follow_up_rationale": (
        "Preempt-structural KILL does NOT spawn an _impl companion (per "
        "F#687/F#698/F#699/F#700/F#701/F#703/F#705/F#710/F#714 precedent + "
        "reviewer.md section 5). Unblock is pre-registration-external "
        "(requires editing DB entry to add target-pair KC + references + "
        "platform), not implementation-external."
    ),
    "drain_subcase_taxonomy": {
        "primary_sub_case": "F#666-pure standalone, routing-accuracy sub-flavor",
        "primary_occurrence_index_in_sub_flavor": 3,
        "secondary_sub_case": "F#666-pure standalone, infrastructure-benchmark sub-flavor",
        "secondary_occurrence_index_in_sub_flavor": 2,
        "orthogonal_to": "F#669 family (parent-unverified)",
        "depends_on": [],
        "approx_overall_index": 23,
        "prior_routing_acc_instances": [
            "F#703 (exp_followup_tfidf_medical_unaliased)",
            "F#710 (exp_g4_gumbel_top2_n50)",
        ],
        "prior_infrastructure_benchmark_instances": [
            "F#734 (exp_composition_clustering_group, K-component latency)",
        ],
    },
    "notes": (
        "Preempt-structural KILL -- no model loaded, no dataset opened, "
        "no tokenizer invoked, no routing classifier trained, no "
        "head-to-head benchmark executed. See MATH.md for theorem + "
        "4-cell verdict truth table + unblock condition (re-register "
        "exp_routing_latency_benchmark_all_behavioral with target-pair KC "
        "and references)."
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
