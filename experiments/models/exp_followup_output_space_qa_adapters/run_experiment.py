"""exp_followup_output_space_qa_adapters — PREEMPTIVE KILL stub.

Not executed. See MATH.md for the three-lemma proof that K1552 is either
tautological (L1), prerequisite-gate unmet (L2), or base-beat-impossible (L3).

No runnable code. Running this file writes a results.json marker and exits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def main() -> None:
    marker = {
        "status": "KILLED",
        "verdict": "KILLED",
        "preemptive": True,
        "executed": False,
        "reason": "L1 tautological-KC OR L2 prerequisite-gate-unmet OR L3 base-beat-impossible",
        "see": "MATH.md",
    }
    (HERE / "results_run_marker.json").write_text(json.dumps(marker, indent=2))
    print("Preemptive kill — no run. See MATH.md and results.json.")
    sys.exit(0)


if __name__ == "__main__":
    main()
