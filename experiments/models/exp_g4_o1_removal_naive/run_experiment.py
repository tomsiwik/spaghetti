"""F#666-pure standalone preempt-KILL scaffold.

Imports `json` + `pathlib` only — no MLX, no training, no inference. `main()`
never raises. Writes `results.json` with `verdict="KILLED"`, KC `untested`,
preempt-reason `F666_PURE_PREEMPT_KILL`. Follows F#700/F#701/F#703 scaffold.
"""
import json
from pathlib import Path


def main() -> None:
    results = {
        "experiment_id": "exp_g4_o1_removal_naive",
        "verdict": "KILLED",
        "all_pass": False,
        "is_smoke": False,
        "preempt_reason": "F666_PURE_PREEMPT_KILL",
        "governing_finding": "F#666",
        "precedent_findings": ["F#700", "F#701", "F#703"],
        "reviewer_md_clause": "§5 KILL (preempt-structural — F#666-pure standalone)",
        "kill_criteria": [
            {
                "id": 1580,
                "text": "max PPL drift <= 0.2% after remove, N=25 -> 24",
                "classification": "proxy",
                "result": "untested",
                "preempt_reason": (
                    "PPL is a proxy metric per guardrail 1007; no paired "
                    "target-metric KC; depends_on: [] (no parent target "
                    "anchor). Per F#666, proxy-PASS-alone is tautological "
                    "SUPPORT and proxy-FAIL-alone is 'a finding about the "
                    "proxy, not a kill'. Both outcome classes unidentifiable."
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
            "Re-register as exp_g4_o1_removal_target_paired with paired "
            "target KC (HumanEval PASS@1 drop <=1pp) + proxy KC (PPL drift "
            "<=0.2%) + neighbor-fidelity KC (>=95% token agreement). See "
            "MATH.md §4 and §8."
        ),
        "impl_followup_filed": False,
        "impl_followup_rationale": (
            "Preempt-structural KILL excludes _impl per F#687/F#698/F#699/"
            "F#700/F#701/F#703 precedent. Unblock is pre-reg-external."
        ),
    }

    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {out}")
    print("verdict=KILLED preempt_reason=F666_PURE_PREEMPT_KILL (no compute)")


if __name__ == "__main__":
    main()
