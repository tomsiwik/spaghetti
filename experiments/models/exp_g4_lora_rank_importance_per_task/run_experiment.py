"""run_experiment.py — exp_g4_lora_rank_importance_per_task (PREEMPT-KILL placeholder)

This experiment is preempt-killed before any code path executes. The pre-registered
KC set {K1941, K1942} is proxy-only with no behavioral target pair, while the
experiment is standalone (depends_on=[]). Per Finding #666 + guardrail 1007
(TARGET-GATED KILL), KILL on a proxy-only KC set is forbidden — a "PASS" outcome
does not certify behavioral benefit and a "FAIL" outcome does not certify
behavioral loss. Both decision branches are unidentifiable as findings.

This file exists for filesystem-conformance only (reviewer.md §1 required-artifacts
checklist). It MUST NOT be executed; running it raises SystemExit immediately.

See MATH.md §1 for the F#666-pure preempt theorem, §4 for the unblock condition
(target-paired re-register as `exp_g4_lora_rank_importance_per_task_v2` or subsume
into `exp_g4_adapter_class_composition_full` K4 once parent SUPPORTED).
"""

import sys


def main() -> int:
    print(
        "PREEMPT-KILL: exp_g4_lora_rank_importance_per_task is structurally blocked per "
        "F#666 + guardrail 1007 (TARGET-GATED KILL). Both KCs are proxies (rank-uniformity "
        "and rank-variance-ratio of argmax_r M(r, task) across tasks) with no behavioral "
        "target pair, on a standalone experiment (depends_on=[]). The 4-cell KC truth "
        "table contains 1 contradictory cell (PASS+PASS impossible: K1941 uniform vs "
        "K1942 variance > 4×), 2 inconclusive cells, and 1 'proxy-PASS without target — "
        "F#666 forbidden' cell. No reachable cell produces a behaviorally-anchored "
        "finding. See MATH.md §4 for the unblock condition.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
