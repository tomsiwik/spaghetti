#!/usr/bin/env python3
"""
exp_sigreg_composition_monitor — SIGReg on N-composition axis.

Graceful-failure stub. Design is locked in MATH.md; empirical verification
requires ≥6 trained Gemma 4 E4B v_proj+o_proj r=6 adapters (currently 0 on
disk per F#627) plus a composition/eval harness whose total budget exceeds
the 30-minute researcher-hat cap by 28-40x.

This file writes a PROVISIONAL results.json with verdict=PROVISIONAL and
all KCs marked not_measured, so downstream review sees an honest state
rather than a silently-upgraded run.

Skills invoked at design stage: /mlx-dev, /fast-mlx (declared in MATH.md §0).
"""
from __future__ import annotations

import json
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"


def main() -> None:
    results = {
        "is_smoke": False,
        "verdict": "PROVISIONAL",
        "all_pass": False,
        "kc": {
            "K1779_spearman_r_gt_0_5": "not_measured",
            "K1780_sigreg_leads_task_acc_by_10pct": "not_measured",
            "K1781_fpr_lt_10pct_at_N1": "not_measured",
        },
        "blockers": [
            "adapter_inventory: 0 of >=6 required v_proj+o_proj r=6 adapters on disk "
            "(F#627 trained q_proj, not v_proj+o_proj; training 24 adapters = ~10h)",
            "composition_harness: not implemented (W_comp = W + sum_i Delta_i at N in {1,2,4,8,16,24})",
            "sigreg_capture: not implemented (layer-21 hook + M=1024 projections + Epps-Pulley K=32 quadrature)",
            "task_accuracy_grid: not implemented (A(N) per |S| on fixed prompt set)",
            "correlation_and_leadtime: not implemented (Spearman + lead-time vs training-step)",
        ],
        "rationale": (
            "Design-locked PROVISIONAL per F#682/F#691 precedent. "
            "Novel axis: N-composition (vs F#682 layer, F#691 depth). "
            "Full empirical budget ~14-20h wall-clock, exceeds researcher-hat 30-min cap "
            "by 28-40x. Deferred to _impl claim."
        ),
        "notes": "See MATH.md §7 for _impl TODO checklist; PAPER.md for verdict table.",
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print("verdict: PROVISIONAL — design-locked, empirical deferred (see MATH.md §6–§7)")


if __name__ == "__main__":
    main()
