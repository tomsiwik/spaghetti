"""exp_followup_routing_output_space_top2 — PREEMPTIVE KILL stub.

Not executed. See MATH.md for the five-lemma proof that K1577 is either
tautological (L1), prerequisite-gate unmet (L2), base-beat-impossible (L3),
bundled-fixes-unidentifiable (L4), or a duplicate of already-killed
exp_followup_output_space_qa_adapters K1552 (L5).

No runnable code. Running this file writes results.json and exits 0.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

RESULTS = {
    "experiment_id": "exp_followup_routing_output_space_top2",
    "status": "KILLED",
    "verdict": "KILLED",
    "preemptive": True,
    "executed": False,
    "is_smoke": False,
    "all_pass": False,
    "preempt_reason": "TAUTOLOGICAL_INTER_ADAPTER_DELTA_IGNORES_BASE_BASELINE",
    "kill_criteria": [
        {
            "id": 1577,
            "text": "QA-format + cache-aware top-2 beats NTP+swap-per-token baseline on QA by >=5pp",
            "result": "untested",
            "rationale": "Structurally uninformative KC. See MATH.md §1 L1-L5.",
        },
    ],
    "findings_reused": [
        {"id": 165, "use": "Output-space avg of individually-harmful adapters bounded below base (-24% on Falcon-E-3B)"},
        {"id": 166, "use": "Prerequisite gate: single adapter must beat base before composition is testable"},
        {"id": 167, "use": "Runtime LoRA IS output-space MoE; binding constraint is base quality"},
        {"id": 168, "use": "Cross-terms structurally impossible in output-space LoRA composition (LoRI)"},
        {"id": 477, "use": "Gemma 4 single-adapter base-beat rate 2/5; Falcon-E-3B not stronger; gate structurally unlikely"},
    ],
    "sibling_precedent": {
        "experiment_id": "exp_followup_output_space_qa_adapters",
        "kc_id": 1552,
        "status": "killed",
        "killed_on": "2026-04-19",
        "relation": "textually-equivalent-duplicate",
        "note": "K1552 and K1577 differ only in baseline naming; same parent-kill motivation; L1-L4 preempt applies byte-for-byte",
    },
    "antipattern_flags": [
        "tautological-inter-adapter-delta-ignores-base-baseline",
        "prerequisite-gate-unmet-output-space-composition",
        "bundled-orthogonal-fixes-format-plus-speed-one-kc",
        "format-alignment-symptom-fix-not-disease",
        "duplicate-of-already-killed-pre-reg",
    ],
    "no_rerun_justification": (
        "A valid v2 requires (a) pre-registered base-beat gate, (b) base-anchored KC "
        "(composition vs base not vs straw NTP variant), (c) decomposed format and "
        "cache-aware fixes into separate KCs. This pre-reg would need redesign with a "
        "new experiment ID, not a re-run."
    ),
    "behavioral_predictions_not_measured": {
        "Q_A_NTP_MMLU": "<= 0.42 (reproduces F#165)",
        "Q_A_QA_MMLU": "[0.47, 0.53] (format-fix lifts into MCQ regime, not over base)",
        "K1577_delta": ">= 5pp (tautological per L1)",
        "Q_A_QA_minus_Q_base": "[-0.08, -0.01] (adapters still lag base per F#477)",
    },
}


def main() -> None:
    (HERE / "results.json").write_text(json.dumps(RESULTS, indent=2) + "\n")
    print("Preemptive kill — no run. See MATH.md and results.json.")


if __name__ == "__main__":
    main()
