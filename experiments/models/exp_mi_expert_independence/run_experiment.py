"""exp_mi_expert_independence — derivation-only stub.

PREEMPTIVELY KILLED — no model run. See MATH.md for proof:
- K#418 FAIL by Data-Processing-Inequality bound
  r²(MI, behavioral-Q) ≤ r²(PPL, Q) ≈ 0.0064 ≪ 0.1.
- K#419 FAIL by KSG estimator complexity:
  C_MI / C_cos ≥ d·log(n)·k_const ≥ 4.96e5 at d=2304, n=75 ≫ 100.

Recycles F#22 (KL killed, "same failure as cosine gating"), F#544
(KL ρ=−0.7 vs quality at N=5 macro), F#286 (PPL/behavior r=0.08).

This file exists only so `experiment complete` finds the canonical
runner name. Running it emits the kill rationale and writes
results.json with all_pass=false.
"""

import json
import sys
from pathlib import Path

KILL_RATIONALE = {
    "verdict": "KILLED",
    "all_pass": False,
    "is_smoke": False,
    "run_type": "derivation-only",
    "kill_criteria": {
        "K418": {
            "text": "MI doesn't predict composition quality better than cosine (r² improvement <0.1)",
            "result": "fail",
            "evidence": (
                "DPI bound: r²(MI,Q) ≤ r²(distributional-channel,Q) ≤ "
                "r²(PPL,Q) = 0.0064 (F#286 r=0.08). MI improvement over "
                "cosine bounded by 0.0064 ≪ 0.1. F#22 (killed) explicitly "
                "kills KL on same ground; MI inherits via MI = KL(p(X,Y)‖p(X)p(Y))."
            ),
        },
        "K419": {
            "text": "MI computation cost >100x cosine computation cost",
            "result": "fail",
            "evidence": (
                "KSG estimator: C_MI/C_cos ≥ d·log(n)·k_const = "
                "2304·4.3·50 ≈ 4.96e5 for d=2304, n=75. ≫ 100. "
                "MINE neural estimator strictly costlier."
            ),
        },
    },
    "preempt_axis": "composition-bug/parent-finding-contradicts-assumption",
    "preempt_subvariant": "distributional-metric-on-proxy-channel",
    "parent_findings": [22, 544, 286, 285, 425],
}


def main() -> int:
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(KILL_RATIONALE, indent=2))
    print(json.dumps(KILL_RATIONALE, indent=2))
    return 1  # non-zero — KILLED


if __name__ == "__main__":
    sys.exit(main())
