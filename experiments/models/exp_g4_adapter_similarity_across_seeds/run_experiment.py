"""run_experiment.py — exp_g4_adapter_similarity_across_seeds

PREEMPT-KILL stub. No execution. See MATH.md for the structural theorem.

This file exists to satisfy the experiment-artifact contract (every experiment dir
must contain run_experiment.py). It is intentionally inert: F#666-pure standalone
preempt-structural KILL means no model is loaded, no dataset is opened, no
training loop runs, no adapters trained, no pairwise cos-sim computed.

Verdict is determined from the KC-set shape:
  K = {K1937 (cos > 0.80 deterministic-tail proxy), K1938 (cos < 0.30 seed-dependent-tail proxy)}
  Both are dual-tail thresholds on the same proxy quantity (cross-instance pairwise
  cos-sim population statistic). No target-metric KC. Under F#666, no PASS/FAIL
  combination yields a valid KILL or SUPPORTED verdict; the dual-tail design admits
  a 3-cell verdict map (deterministic-tail / intermediate / seed-dependent-tail),
  all three inadmissible.

To re-register a runnable version, see MATH.md §5 unblock conditions.
"""

import json
import sys
from pathlib import Path

VERDICT = "KILLED"
PREEMPT_REASON = (
    "F666_PURE_STANDALONE_PROXY_ONLY_KC_SET_G4_ABLATION_SEED_DETERMINISM_1ST_SUBTYPE"
)


def main() -> int:
    out = Path(__file__).parent / "results.json"
    if not out.exists():
        print(
            "results.json missing — preempt-KILL artifact set incomplete; see MATH.md",
            file=sys.stderr,
        )
        return 2
    payload = json.loads(out.read_text())
    if payload.get("verdict") != VERDICT:
        print(
            f"results.json verdict mismatch: expected {VERDICT}, got {payload.get('verdict')}",
            file=sys.stderr,
        )
        return 2
    print(f"PREEMPT-KILL — {PREEMPT_REASON}. See MATH.md / PAPER.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
