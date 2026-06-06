"""run_experiment.py — exp_sigreg_threshold_sweep

PREEMPT-STRUCTURAL-KILL. No computation is performed.

Three independent guardrails each suffice to block this experiment — see MATH.md
§1.1 (F#666-pure, both KCs are proxy), §1.2 (§5 intra-detector-threshold-delta,
2nd intra-instantiation sub-variant), §1.3 (F#669 parent-target-unverified,
parent F#713 PROVISIONAL).

This scaffold exists only to satisfy the platform hygiene guardrail (F#702): every
experiment dir must contain a runnable `run_experiment.py` that writes
`results.json`. It emits a results.json documenting the preempt-structural KILL.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    results = {
        "experiment_id": "exp_sigreg_threshold_sweep",
        "verdict": "KILLED",
        "kill_type": "preempt-structural",
        "all_pass": False,
        "is_smoke": False,
        "preempt_memories_fired": [
            "mem-antipattern-f666-pure-standalone-preempt-kill",
            "mem-antipattern-tautological-inter-variant-delta",
            "mem-promotion-same-parent-repeat-blocker",  # F#669 family generalization
        ],
        "f666_pure_index": 18,
        "paragraph5_index": 12,
        "paragraph5_sub_variant": "intra-detector-threshold-delta",
        "paragraph5_intra_instantiation_instance": 2,  # after F#712 intra-adapter-rank-delta
        "f669_reuse_index": 8,
        "triple_fire_index": 8,
        "triple_fire_sub_composition": "f666-pure+paragraph5+f669",
        "cross_parent_instance": 1,  # first with non-F#682 parent (parent is F#713)
        "parent_experiment": "exp_sigreg_composition_monitor",
        "parent_finding": 713,
        "parent_status": "PROVISIONAL",
        "wall_clock_seconds": 0.0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kcs": [
            {"id": 1890, "result": "inconclusive", "type": "proxy",
             "reason": "classification-accuracy proxy (FPR); no target companion"},
            {"id": 1891, "result": "inconclusive", "type": "proxy",
             "reason": "classification-accuracy proxy (FNR); no target companion"},
        ],
        "notes": "See MATH.md for the three independent structural proofs.",
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"[exp_sigreg_threshold_sweep] preempt-structural KILL written to {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
