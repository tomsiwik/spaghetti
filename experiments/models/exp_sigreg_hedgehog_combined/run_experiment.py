"""exp_sigreg_hedgehog_combined — PREEMPT-KILL stub.

Structural preempt-KILL on F#666-pure-standalone (10th instance, multi-bucket
across F#711-refactored taxonomy: K1854 = derived-geometric (cos-sim),
K1855 = detection (Epps-Pulley statistic)). Secondary fire: §5
tautological-inter-variant-delta (5th instance, 2nd inter-training axis after
F#704). Tertiary fire: hygiene-multi-defect (3+ defects: success_criteria=[],
references=[], platform=~) — but hygiene-patch path (F#702) unavailable because
zero target-metric KCs exist to patch around. No measurement performed. See
MATH.md for the full proof.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    exp_dir = Path(__file__).parent
    results = {
        "experiment_id": "exp_sigreg_hedgehog_combined",
        "verdict": "KILLED",
        "all_pass": False,
        "is_smoke": False,
        "preempt_structural": True,
        "antipattern_primary": "f666-pure-standalone",
        "antipattern_primary_instance": 10,
        "antipattern_primary_buckets": ["derived-geometric", "detection"],
        "antipattern_secondary": "tautological-inter-variant-delta-ignores-base-baseline",
        "antipattern_secondary_instance": 5,
        "antipattern_secondary_axis": "inter-training-method (2nd; 1st was F#704)",
        "antipattern_tertiary": "hygiene-multi-defect-3plus",
        "antipattern_tertiary_defects": [
            "success_criteria=[]",
            "references=[]",
            "platform=~",
        ],
        "hygiene_patch_unavailable": (
            "F#702 hygiene-patch path requires >=1 target-metric KC. Both KCs are "
            "proxy-only (cos-sim, SIGReg-statistic). F#666-pure preempts."
        ),
        "clause": "guardrail #1007 / F#666 canonical (target-gated KILL)",
        "kill_criteria": [
            {
                "id": 1854,
                "text": (
                    "Combined loss Hedgehog adapter cos-sim > 0.05 worse than "
                    "Hedgehog-only (SIGReg hurts)"
                ),
                "result": "UNMEASURABLE",
                "kc_class": "proxy-only (cos-sim similarity, derived-geometric bucket)",
                "note": (
                    "Comparison-only KC without per-variant base-anchor (also fires "
                    "§5 inter-training axis). Cos-sim is structural similarity proxy; "
                    "F#688 measured r=0.08 between proxy and task quality on this "
                    "codebase. Both PASS and FAIL branches are F#666-inadmissible: "
                    "PASS=tautological-support, FAIL=finding-about-proxy-not-kill."
                ),
            },
            {
                "id": 1855,
                "text": (
                    "SIGReg statistic during training shows collapse at any checkpoint"
                ),
                "result": "UNMEASURABLE",
                "kc_class": "proxy-only (collapse-detection, detection bucket)",
                "note": (
                    "Epps-Pulley statistic on training-time hidden-state moments. No "
                    "paired downstream task accuracy. Both branches F#666-inadmissible: "
                    "PASS=tautological-support (non-rejection != task gain), "
                    "FAIL=finding-about-proxy (collapse during training does not imply "
                    "downstream task failure; Pierre v3 achieves 0.41 behavioral "
                    "despite collapse-prone hidden-state distributions)."
                ),
            },
        ],
        "lemmas": [
            "L1 both KCs proxy-only under F#666 / guardrail #1007 (multi-bucket: derived-geometric K1854 + detection K1855)",
            "L2 F#666 truth-table inadmissibility (PASS=tautological-support, FAIL=finding-about-proxy on both KCs)",
            "L3 §5 secondary fire (5th instance, 2nd inter-training axis after F#704)",
            "L4 hygiene-multi-defect tertiary (3+ defects); F#702 hygiene-patch path unavailable (zero target KCs)",
            "L5 standalone topology (depends_on=[], blocks=[]); not F#669, not template-regression, not proxy-only-lineage-inheritance",
        ],
        "unblock_path": (
            "v2 exp_sigreg_hedgehog_combined_v2_target_gated: pair each proxy KC with "
            "a target KC (task_acc deltas with base-anchor + per-variant floor per "
            "F#166), restrict cos-sim/SIGReg-statistic to diagnostic role, hygiene "
            "patch (success_criteria, references, platform=local-apple). Cite F#666, "
            "F#688, F#682/691/713 SIGReg triad, F#627 Hedgehog parent-class."
        ),
        "references": [
            "F#666 (conclusive 2026-04-19) — target-gate guardrail",
            "F#477 (killed 2026-04-11) — Gemma 4 shallow-regime decoupling",
            "F#688 — measured PPL<->task r=0.08 on this codebase",
            "F#704 (killed 2026-04-24) — §5 2nd instance, inter-training 1st",
            "F#709 (killed 2026-04-24) — §5 3rd instance, §5 promotion",
            "F#712 (killed 2026-04-24) — §5 4th instance, intra-rank sub-variant",
            "F#700, F#701, F#703, F#705, F#706, F#707, F#708, F#710, F#711 — F#666-pure 1-9",
            "F#711 — taxonomy refactor execution (multi-bucket fire here)",
            "F#702 (provisional 2026-04-24) — hygiene-patch path (N/A here)",
            "F#703 — hygiene-multi-defect 3+ canonical",
            "F#682, F#691, F#713 — SIGReg triad design-locks",
            "F#627 — Hedgehog-class r=6 adapter SUPPORTED on domain tasks",
            "F#166 — prerequisite-gate (base-beat before delta)",
            "LeWM arxiv:2603.19312 — SIGReg/Epps-Pulley grounding",
            "Hedgehog arxiv:2402.04347 — cos-sim distillation grounding",
        ],
    }
    out = exp_dir / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
