"""run_experiment.py — exp_hedgehog_early_stopping_cos_plateau

PREEMPT-KILL stub. No execution. See MATH.md for the structural theorem.

This file exists to satisfy the experiment-artifact contract (every experiment dir
must contain run_experiment.py). It is intentionally inert: F#666-pure standalone
preempt-structural KILL means no model is loaded, no dataset is opened, no
training loop runs, no per-layer cos-sim is computed, no plateau detector fires.

Verdict is determined from the KC-set shape:
  K = {K1935 (cos-sim tightness proxy), K1936 (training-time-savings proxy)}
  No target-metric KC. Under F#666, no PASS/FAIL combination yields a valid
  KILL or SUPPORTED verdict.

To re-register a runnable version, see MATH.md §5 unblock conditions.
"""

import json
import sys
from pathlib import Path

VERDICT = "KILLED"
PREEMPT_REASON = (
    "F666_PURE_STANDALONE_PROXY_ONLY_KC_SET_HEDGEHOG_ABLATION_7TH_SUBTYPE"
)


def main() -> int:
    out = Path(__file__).parent / "results.json"
    if not out.exists():
        # results.json is authored alongside MATH.md, not generated here.
        # The presence-check is purely defensive.
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
