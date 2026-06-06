"""F#666-pure standalone preempt-KILL scaffold.

Imports `json` + `pathlib` only — no MLX, no training, no inference, no routing.
`main()` never raises. Writes `results.json` with `verdict="KILLED"`, KC
`untested`, preempt-reason `F666_PURE_PREEMPT_KILL`. Follows
F#700/F#701/F#703/F#705/F#706/F#707 scaffold.
"""
import json
from pathlib import Path


def main() -> None:
    results = {
        "experiment_id": "exp_g4_hash_ring_remove_n25",
        "verdict": "KILLED",
        "all_pass": False,
        "is_smoke": False,
        "preempt_reason": "F666_PURE_PREEMPT_KILL",
        "governing_finding": "F#666",
        "precedent_findings": ["F#700", "F#701", "F#703", "F#705", "F#706", "F#707"],
        "reviewer_md_clause": "§5 KILL (preempt-structural — F#666-pure standalone)",
        "drain_window_row": 7,
        "ppl_proxy_instance": 2,
        "sub_pattern_candidate": "template-regression (2nd instance: parent F#133 paired-KC design stripped to PPL-only)",
        "kill_criteria": [
            {
                "id": 1583,
                "text": "mean PPL <= 3%, max <= 5%",
                "classification": "proxy",
                "result": "untested",
                "preempt_reason": (
                    "PPL is a proxy metric per guardrail 1007; both sub-thresholds "
                    "(mean ≤3%, max ≤5%) collapse to one PPL-axis verdict. No paired "
                    "target-metric KC; depends_on: [] (no parent target anchor). Per "
                    "F#666, proxy-PASS-alone is tautological SUPPORT and proxy-FAIL-"
                    "alone is 'a finding about the proxy, not a kill'. Both outcome "
                    "classes unidentifiable. Notable: parent F#133 itself uses PAIRED "
                    "KC design (K1 PPL + K2 neighbor accuracy) — child stripped K2."
                ),
            }
        ],
        "success_criteria": [],
        "hygiene_defects": {
            "success_criteria_empty": True,
            "references_empty_in_field": True,
            "platform_missing": False,
            "count": 2,
            "below_3plus_threshold": True,
        },
        "unblock_path": (
            "Re-register as exp_g4_hash_ring_remove_n25_target_paired with paired "
            "target KCs mirroring parent F#133 design: K1 HumanEval PASS@1 drop ≤1pp "
            "+ K2 neighbor accuracy ≥95% + K3 PPL drop ≤3%/5% (sanity). See MATH.md "
            "§4 and §8."
        ),
        "impl_followup_filed": False,
        "impl_followup_rationale": (
            "Preempt-structural KILL excludes _impl per F#687/F#698/F#699/F#700/"
            "F#701/F#703/F#705/F#706/F#707 precedent. Unblock is pre-reg-external."
        ),
    }

    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {out}")
    print("verdict=KILLED preempt_reason=F666_PURE_PREEMPT_KILL (no compute)")


if __name__ == "__main__":
    main()
