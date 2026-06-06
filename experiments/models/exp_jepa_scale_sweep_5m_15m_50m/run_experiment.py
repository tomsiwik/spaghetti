"""exp_jepa_scale_sweep_5m_15m_50m — preempt-structural KILL (F#669-family).

No measurement. F#669-family clause: child KCs reference parent measurements
(K1768 GSM8K-Hard baseline) that the parent has not produced. Parent
exp_jepa_adapter_residual_stream is PROVISIONAL (F#682), all 4 KCs untested.
Parent _impl exp_jepa_adapter_residual_stream_impl is PROVISIONAL (F#772),
Phase A token-space LoRA only; Phase B/C custom MLX training loop
NotImplementedError.

Schema-repair (F#770 cohort fix iter ~38) added paired target KCs K1988/K1989
to satisfy F#666-discipline. Repair was necessary but not sufficient — F#666
gates KC structure; F#669 gates threshold measurability. The repair migrated
the diagnosis from F#666-pure-standalone (KC-design-defect) to F#669-cascade
(parent-cascade-defect). 2nd observation of the schema-repair-reveals-F#669
meta-pattern (F#776 promoting from 1st-obs).

See MATH.md §1 for the preempt theorem and §4 for the unblock condition.
PAPER.md carries the prediction-vs-measurement table; results.json carries
the canonical preempt-KILL payload.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

PAYLOAD = {
    "experiment_id": "exp_jepa_scale_sweep_5m_15m_50m",
    "verdict": "KILLED",
    "verdict_reason": "preempt-structural (F#669-family clause)",
    "all_pass": False,
    "is_smoke": False,
    "measurements_taken": 0,
    "kill_criteria": [
        {
            "id": 1862,
            "text": "Next-embedding MSE doesn't improve from 5M->15M (diminishing returns at small scale)",
            "kind": "proxy",
            "result": "untested",
            "reason": "preempt-blocked (F#669): paired target K1988 references parent K1768 (untested); proxy alone insufficient per F#666.",
        },
        {
            "id": 1863,
            "text": "15M adapter training > 90 min on M5 Pro (scale ceiling)",
            "kind": "proxy",
            "result": "untested",
            "reason": "preempt-blocked (F#669): paired target K1989 requires per-scale trained JEPA adapters; parent _impl Phase B NotImplementedError; proxy alone insufficient per F#666.",
        },
        {
            "id": 1988,
            "text": "best-scale (5M/15M/50M) JEPA adapter GSM8K-Hard accuracy >= token-space r=16 LoRA baseline at matched compute on Gemma 4 E4B, n>=100, greedy",
            "kind": "target",
            "result": "untested",
            "reason": "preempt-blocked (F#669): RHS = token-space r=16 LoRA baseline (parent K1768 measurement); parent K1768 untested => RHS = NaN => comparison unidentifiable. F#770-schema-repair-target added 2026-04-25; explicitly inherits parent K1768 target.",
        },
        {
            "id": 1989,
            "text": "best-scale GSM8K-Hard accuracy beats worst-scale by >= 2pp on n>=100 OR all 3 scales within 1pp behaviorally (compute-optimal scale exists OR saturation finding)",
            "kind": "target",
            "result": "untested",
            "reason": "preempt-blocked (F#669): per-scale GSM8K-Hard accuracy requires trained JEPA residual-stream adapter at each of {5M, 15M, 50M}; parent _impl Phase B custom MLX training loop NotImplementedError; no executable JEPA training mechanism exists. Drop undefined for all 3 scales simultaneously. F#770-schema-repair-target added 2026-04-25.",
        },
    ],
    "preempt_basis": {
        "clause": "F#669-family preempt-structural",
        "f669_reuse_index": 15,
        "f682_child_f669_index": 4,
        "schema_repair_reveals_f669_obs_index": 2,
        "parent_experiment": "exp_jepa_adapter_residual_stream",
        "parent_status_at_claim": "provisional",
        "parent_finding": "F#682 (PROVISIONAL design-only, all 4 KCs K1766/K1767/K1768/K1769 untested)",
        "parent_impl_experiment": "exp_jepa_adapter_residual_stream_impl",
        "parent_impl_status_at_claim": "provisional",
        "parent_impl_finding": "F#772 (PROVISIONAL Phase A only: 50-step token-space LoRA val loss 1.840->0.581, n=10 baseline=40.0%; Phase B/C NotImplementedError)",
        "schema_repair_history": {
            "iter_38_f770_repair": "added paired target KCs K1988/K1989 to satisfy F#666-discipline",
            "diagnosis_migration": "F#666-pure-standalone (pre-repair, KC-design-defect) -> F#669-cascade (post-repair, parent-cascade-defect)",
            "meta_pattern": "schema-repair-reveals-F#669 — 2nd observation; 1st obs F#776 (rank_ablation iter ~48); promotion at 3rd obs per mem-pattern-triple-fire",
            "cross_cluster_confirmation": "Hedgehog cluster (F#776) + JEPA cluster (this) — meta-pattern is NOT cluster-specific",
            "predicted_3rd_instance": "exp_hedgehog_cross_axis_interference IF F#770-repaired (currently F#666-pure standalone) OR another cohort child as F#770 cohort drains",
        },
        "f682_cluster_history": [
            "F#727 (jepa_multilayer_prediction, 1st F#682-child F#669, F#669 5th reuse, pre-F#770)",
            "F#728 (jepa_contrastive_variant, 2nd F#682-child F#669, pre-F#770)",
            "F#729 (jepa_frozen_encoder_ablation, 3rd F#682-child F#669, pre-F#770)",
            "this is 4th F#682-child F#669 instance, AND 1st post-F#770 F#682-child F#669",
        ],
        "drain_window_index": 49,
    },
    "platform_skills_invoked": [
        "/mlx-dev (noted, not used — no code path; F#669-family clause exempts skill-attestation gate)",
        "/fast-mlx (noted, not used — no code path)",
    ],
    "base_model": "mlx-community/gemma-4-e4b-it-4bit (per F#627, not loaded)",
    "impl_follow_up_filed": False,
    "impl_follow_up_rationale": "Preempt-structural KILL does not spawn an _impl companion (per F#669-family clause + reviewer.md §5). Unblock is parent-external: exp_jepa_adapter_residual_stream_impl (P=1, currently `provisional` per F#772 with Phase B/C NotImplementedError) is the unblock gate; this child is downstream of parent's _impl Phase B/C SUPPORTED transition.",
}


def main() -> None:
    out = HERE / "results.json"
    out.write_text(json.dumps(PAYLOAD, indent=2) + "\n")
    print(json.dumps(PAYLOAD, indent=2))


if __name__ == "__main__":
    main()
