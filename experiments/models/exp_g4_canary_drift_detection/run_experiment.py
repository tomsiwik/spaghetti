"""F#666-pure standalone preempt-KILL scaffold.

Imports `json` + `pathlib` only — no MLX, no training, no inference. `main()`
never raises. Writes `results.json` with `verdict="KILLED"`, KC `untested`,
preempt-reason `F666_PURE_PREEMPT_KILL`. Follows F#700/F#701/F#703/F#705 scaffold.
"""
import json
from pathlib import Path


def main() -> None:
    results = {
        "experiment_id": "exp_g4_canary_drift_detection",
        "verdict": "KILLED",
        "all_pass": False,
        "is_smoke": False,
        "preempt_reason": "F666_PURE_PREEMPT_KILL",
        "governing_finding": "F#666",
        "precedent_findings": ["F#700", "F#701", "F#703", "F#705"],
        "reviewer_md_clause": "§5 KILL (preempt-structural — F#666-pure standalone)",
        "kill_criteria": [
            {
                "id": 1581,
                "text": "FNR <= 5% on synthetic-corrupted adapter",
                "classification": "proxy",
                "proxy_flavor": "classification_accuracy_on_synthetic_detection_test",
                "result": "untested",
                "preempt_reason": (
                    "FNR is classification-accuracy per guardrail 1007 explicit "
                    "enumeration; synthetic-corrupted-adapter test set is a proxy "
                    "for the behavioral target (detection of compositions that "
                    "actually degrade user-visible task accuracy). No paired "
                    "target-metric KC (task-accuracy TPR on degrading comps, FPR "
                    "on safe comps, mechanistic rho*cos correlation). "
                    "depends_on: [] — parent F#156's mechanistic linkage is not "
                    "inherited operationally. Per F#666: proxy-PASS-alone is "
                    "tautological SUPPORT and proxy-FAIL-alone is 'a finding "
                    "about the proxy, not a kill'. Both outcome classes "
                    "unidentifiable."
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
            "Re-register as exp_g4_canary_drift_target_paired with paired "
            "target KCs: K1 TPR>=95% on degrading comps, K2 FPR<=10% on safe "
            "comps, K3 synthetic-FNR<=5% (sanity), K4 rho*cos correlation>=0.5 "
            "(mechanistic anchor). See MATH.md §4 and §8."
        ),
        "impl_followup_filed": False,
        "impl_followup_rationale": (
            "Preempt-structural KILL excludes _impl per F#687/F#698/F#699/"
            "F#700/F#701/F#703/F#705 precedent. Unblock is pre-reg-external."
        ),
    }

    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {out}")
    print("verdict=KILLED preempt_reason=F666_PURE_PREEMPT_KILL (no compute)")


if __name__ == "__main__":
    main()
