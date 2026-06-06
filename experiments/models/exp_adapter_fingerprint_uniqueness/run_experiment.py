"""run_experiment.py — exp_adapter_fingerprint_uniqueness (PREEMPT-KILL placeholder)

This experiment is preempt-killed before any code path executes. The pre-registered
KC set {K1943, K1944} is engineering-primitive-only (collision-rate + per-adapter
hash latency) with no behavioral target pair, while the experiment is standalone
(depends_on=[]). Per Finding #666 + guardrail 1007 (TARGET-GATED KILL), KILL on a
KC set that tests only hash-primitive properties — without anchoring to Pierre's
behavioral fingerprint use (versioning / dedup / cache-key correctness under real
workflows) — is forbidden. PASS is tautological for any commodity hash (SHA-256
birthday bound at N=1000 gives collision probability ~4.3e-72); FAIL is an
implementation or library-selection defect, not a research claim.

This file exists for filesystem-conformance only (reviewer.md §1 required-artifacts
checklist). It MUST NOT be executed; running it raises SystemExit immediately.

See MATH.md §1 for the F#666-pure preempt theorem, §4 for the unblock condition
(target-paired re-register as `exp_adapter_fingerprint_uniqueness_v2` with a
behavioral fingerprint-use KC, or subsume into a Pierre-integrated versioning/dedup
experiment).
"""

import sys


def main() -> int:
    print(
        "PREEMPT-KILL: exp_adapter_fingerprint_uniqueness is structurally blocked per "
        "F#666 + guardrail 1007 (TARGET-GATED KILL). Both KCs are engineering-primitive "
        "(K1943 collision-rate + K1944 per-adapter hash-latency) with no behavioral "
        "target pair, on a standalone experiment (depends_on=[]). KC truth-table has "
        "zero behaviorally-anchored cells — all 4 cells resolve as tautology (any "
        "commodity hash trivially passes both) / engineering defect (slow hash) / "
        "implementation defect (truncated-hash or bad canonicalization). No reachable "
        "cell produces a research finding. See MATH.md §4 for the unblock condition.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
