"""exp_g4_gumbel_top2_n50 — preempt-structural KILL stub.

No measurement is performed. The verdict is pre-registered KILLED per MATH.md:
F#666-pure-standalone, 8th drain-window instance, 2nd routing-accuracy sub-flavor
(confirmed-recurrent). The single pre-registered KC (K1591 "acc >= 85%") is a
proxy-only routing-classification threshold with no paired target KC, which
under F#666 / guardrail 1007 admits no compliant verdict:
  - proxy-PASS alone  -> tautological support (forbidden by F#666 canonical)
  - proxy-FAIL alone  -> "finding about proxy, not kill" (forbidden)

This script intentionally writes results.json with verdict=KILLED and exits
without loading any model, tokenizer, or dataset. Re-registration path is
described in MATH.md "Unblock path".
"""

import json
from pathlib import Path

RESULTS = {
    "verdict": "KILLED",
    "preempt_reason": "F666_PURE_STANDALONE_PROXY_ONLY_KC",
    "sub_flavor": "routing_accuracy",
    "sub_flavor_instance": 2,
    "drain_window_instance": 8,
    "all_pass": False,
    "is_smoke": False,
    "kill_criteria": [
        {
            "id": 1591,
            "text": "acc >= 85%",
            "result": "fail",
            "reason": (
                "proxy-only routing-classification KC violates F#666 / "
                "guardrail 1007 — no paired target-metric KC present; "
                "verdict is structurally tautological regardless of measurement"
            ),
        }
    ],
    "evidence": {
        "parent_finding": 72,
        "parent_structure": (
            "pre-F#666 SUPPORTED on 3 proxy-only KCs "
            "(K1 routing-acc, K2 gamma_uniform, K3 max-degradation); "
            "zero target-metric KCs"
        ),
        "canonical_routing_acc_preempt": 703,
        "canonical_f666": 666,
        "guardrail": 1007,
        "drain_window_siblings": [700, 701, 703, 705, 706, 707, 708],
        "refactor_trigger": "live_at_row_5_firm_at_row_8",
    },
    "notes": (
        "Preempt-structural KILL — no model loaded, no dataset opened, "
        "no tokenizer invoked. See MATH.md for theorem + 5 lemmas + "
        "pre-claim 6-item checklist (5 template-regression + "
        "direction-symmetric inter-variant delta)."
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
