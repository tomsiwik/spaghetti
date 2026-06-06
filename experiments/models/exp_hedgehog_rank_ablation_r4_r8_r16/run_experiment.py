"""exp_hedgehog_rank_ablation_r4_r8_r16 — preempt-structural KILL (F#669-family).

No measurement. F#669-family clause: child KCs reference parent measurements
(K1783 ΔJudge, K1784 non-interference) that the parent has not produced.
Parent exp_hedgehog_behavior_adapter_politeness is PROVISIONAL (F#683),
Phase B NotImplementedError, all 4 KCs untested.

Schema-repair (F#770 cohort fix iter ~36) added paired target KCs K1980/K1981/K1982
to satisfy F#666-discipline. Repair was necessary but not sufficient — F#666 gates
KC structure; F#669 gates threshold measurability. The repair migrated the
diagnosis from F#666-pure-standalone (KC-design-defect) to F#669-cascade
(parent-cascade-defect). 1st observation of the schema-repair-reveals-F#669
meta-pattern.

See MATH.md §1 for the preempt theorem and §4 for the unblock condition.
PAPER.md carries the prediction-vs-measurement table; results.json carries
the canonical preempt-KILL payload.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

PAYLOAD = {
    "experiment_id": "exp_hedgehog_rank_ablation_r4_r8_r16",
    "verdict": "KILLED",
    "verdict_reason": "preempt-structural (F#669-family clause)",
    "all_pass": False,
    "is_smoke": False,
    "measurements_taken": 0,
    "kill_criteria": [
        {
            "id": 1852,
            "text": "r=4 cos-sim < 80% of r=8 cos-sim (rank too low for routing capture)",
            "kind": "proxy",
            "result": "untested",
            "reason": "preempt-blocked (F#669): paired target K1980 references parent K1783 (untested); proxy alone insufficient per F#666.",
        },
        {
            "id": 1853,
            "text": "r=16 achieves < 5pp improvement over r=8 (diminishing returns)",
            "kind": "proxy",
            "result": "untested",
            "reason": "preempt-blocked (F#669): paired target K1981 references parent K1783 (untested); proxy alone insufficient per F#666.",
        },
        {
            "id": 1980,
            "text": "r=4 ΔJudge < (r=8 ΔJudge − 5pp) on n=100 neutral prompts using parent K2 rubric",
            "kind": "target",
            "result": "untested",
            "reason": "preempt-blocked (F#669): RHS = (r=8 ΔJudge − 5pp); r=8 ΔJudge is parent K1783 measurement; parent K1783 untested ⇒ RHS = NaN ⇒ comparison unidentifiable. F#770-schema-repair-target added 2026-04-25.",
        },
        {
            "id": 1981,
            "text": "r=16 ΔJudge < (r=8 ΔJudge + 3pp) on n=100 neutral prompts using parent K2 rubric",
            "kind": "target",
            "result": "untested",
            "reason": "preempt-blocked (F#669): RHS = (r=8 ΔJudge + 3pp); same NaN-RHS structure as K1980. F#770-schema-repair-target added 2026-04-25.",
        },
        {
            "id": 1982,
            "text": "For ANY r ∈ {4,8,16}: MMLU-subset acc drop > 3pp OR HumanEval pass@1 drop > 3pp vs base Gemma 4 E4B",
            "kind": "target",
            "result": "untested",
            "reason": "preempt-blocked (F#669): per-rank acc drop requires a trained Hedgehog adapter at each rank; parent's Hedgehog distillation loop has no executable MLX implementation (parent F#683 PROVISIONAL, Phase B NotImplementedError). Without trained adapters, drop is undefined for all 3 ranks simultaneously. F#770-schema-repair-target added 2026-04-25; mirrors parent K1784 design.",
        },
    ],
    "preempt_basis": {
        "clause": "F#669-family preempt-structural",
        "f669_reuse_index": 14,
        "hedgehog_cluster_f669_index": 1,
        "parent_experiment": "exp_hedgehog_behavior_adapter_politeness",
        "parent_status_at_claim": "provisional",
        "parent_finding": "F#683 (PROVISIONAL design-only, Phase B NotImplementedError, all 4 KCs untested)",
        "schema_repair_history": {
            "iter_36_f770_repair": "added paired target KCs K1980/K1981/K1982 to satisfy F#666-discipline",
            "diagnosis_migration": "F#666-pure-standalone (pre-repair, KC-design-defect) → F#669-cascade (post-repair, parent-cascade-defect)",
            "meta_pattern": "schema-repair-reveals-F#669 — 1st observation; promotion at 3rd obs per mem-pattern-triple-fire",
            "predicted_2nd_3rd_instances": [
                "exp_jepa_scale_sweep_5m_15m_50m (K1988/K1989 already F#770-repair-added; references parent residual-stream PROVISIONAL K1768)",
                "exp_hedgehog_cross_axis_interference (same parent politeness PROVISIONAL; KC #1859 currently F#666-pure standalone — would migrate to F#669 if F#770-repaired)",
            ],
        },
        "hedgehog_cluster_history": [
            "F#714/F#716/F#720/F#721/F#722/F#723/F#755/F#756 (8 prior preempts, all F#666-pure-standalone or §5 sub-types)",
            "this is 1st Hedgehog-cluster F#669-family instance",
        ],
        "drain_window_index": 47,
    },
    "platform_skills_invoked": [
        "/mlx-dev (noted, not used — no code path; F#669-family clause exempts skill-attestation gate)",
        "/fast-mlx (noted, not used — no code path)",
    ],
    "base_model": "mlx-community/gemma-4-e4b-it-4bit (per F#627, not loaded)",
    "impl_follow_up_filed": False,
    "impl_follow_up_rationale": "Preempt-structural KILL does not spawn an _impl companion (per F#669-family clause + reviewer.md §5). Unblock is parent-external: exp_hedgehog_behavior_adapter_politeness_impl (P=1, currently `open`) is the unblock gate; this child is downstream of parent's _impl SUPPORTED transition.",
}


def main() -> None:
    out = HERE / "results.json"
    out.write_text(json.dumps(PAYLOAD, indent=2) + "\n")
    print(json.dumps(PAYLOAD, indent=2))


if __name__ == "__main__":
    main()
