"""F#666-pure standalone preempt-KILL scaffold.

Imports `json` + `pathlib` only — no MLX, no hashing, no routing, no inference.
`main()` never raises. Writes `results.json` with `verdict="KILLED"`, KC
`untested`, preempt-reason `F666_PURE_PREEMPT_KILL`. Follows F#700/F#701/F#703/
F#705/F#706 scaffold.
"""
import json
from pathlib import Path


def main() -> None:
    results = {
        "experiment_id": "exp_g4_xxhash_routing_n25",
        "verdict": "KILLED",
        "all_pass": False,
        "is_smoke": False,
        "preempt_reason": "F666_PURE_PREEMPT_KILL",
        "governing_finding": "F#666",
        "precedent_findings": ["F#700", "F#701", "F#703", "F#705", "F#706"],
        "reviewer_md_clause": "§5 KILL (preempt-structural — F#666-pure standalone)",
        "kill_criteria": [
            {
                "id": 1582,
                "text": "R < 2.0 at N=25",
                "classification": "proxy",
                "result": "untested",
                "preempt_reason": (
                    "R (routing-collision-rate vs Welch bound) is the "
                    "mathematical dual of routing-match-rate, explicitly "
                    "named in guardrail 1007. No paired target-metric KC; "
                    "depends_on: [] (no parent target anchor; parent F#147 "
                    "is itself a pure hash-statistics study). Per F#666, "
                    "proxy-PASS-alone is tautological SUPPORT and "
                    "proxy-FAIL-alone is 'a finding about the proxy, not "
                    "a kill'. Both outcome classes unidentifiable."
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
            "Re-register as exp_g4_xxhash_routing_n25_target_paired with "
            "paired target KC (HumanEval PASS@1 drop <=1pp vs oracle) + "
            "proxy KC (R<2.0 Welch) + neighbor-fidelity KC (>=95% token "
            "agreement vs oracle routing). See MATH.md §4 and §8."
        ),
        "impl_followup_filed": False,
        "impl_followup_rationale": (
            "Preempt-structural KILL excludes _impl per F#687/F#698/F#699/"
            "F#700/F#701/F#703/F#705/F#706 precedent. Unblock is "
            "pre-reg-external."
        ),
        "drain_window_row": 6,
        "taxonomic_novelty": (
            "First drain-window instance where proxy is "
            "routing-collision-rate R vs Welch bound. Canonical guardrail "
            "1007 'routing match rate' enumeration (dual). Row 5 was "
            "canonical 'classification accuracy' (FNR). Row 6 is canonical "
            "'routing match rate' (R)."
        ),
    }

    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {out}")
    print("verdict=KILLED preempt_reason=F666_PURE_PREEMPT_KILL (no compute)")


if __name__ == "__main__":
    main()
