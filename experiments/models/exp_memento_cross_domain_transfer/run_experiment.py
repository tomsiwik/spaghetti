"""run_experiment.py — exp_memento_cross_domain_transfer (PREEMPT-KILL).

This experiment is preempt-killed per Finding #669. No MLX code is written
because parent `exp_memento_gemma4_replication` is PROVISIONAL (F#685) and
K1906 requires TWO distinct Gemma-4-MEMENTO checkpoints (one trained on
GSM8K-only, one on MMLU-only) that do not exist. Paper authors released
checkpoints for Qwen3 / Phi-4 / Olmo 3 only, on the pooled OpenMementos
228K-trace dataset — not per-corpus separated. Gemma 4 is not among them at
any training mixture.

Additionally, a cross-training-domain KC requires N=2 independent MEMENTO
training runs of parent's _impl — parent's _impl validates a SINGLE training
mixture (the pooled default). The cross-domain KC is strictly stronger than a
single-mixture measurement and remains preempt-blocked even under P.status =
supported at one training mixture.

K1906 is a single target KC (task accuracy ratio), so the KC set is
F#666-compliant by vacuous quantification (no proxy to pair). This is the 3rd
MEMENTO-cluster child preempt-KILL and the 2nd observation of the broader
multi-parent-run sub-axis (after block_size_ablation — 1st obs, scalar-sweep
variant; this — 2nd obs, categorical cross-training-corpus variant).

This scaffold writes a well-formed `results.json` so downstream tooling
(reviewer, analyst, DB `experiment complete`) sees a valid artifact. No code
path raises: the script always produces a non-empty `results.json` that
encodes the preempt-kill verdict and structurally-untestable KC.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    """Return results dict encoding preempt-KILL.

    No MLX import or call is made. No model is loaded. No measurement runs.
    The verdict is structural: parent target-unverified + cross-training-domain
    KC requires N=2 single-corpus parent _impl runs that do not exist.
    """
    return {
        "experiment_id": "exp_memento_cross_domain_transfer",
        "verdict": "KILLED",
        "kill_reason": (
            "preempt-child-parent-target-unverified + "
            "multi-parent-run-strictly-stronger-than-single-config "
            "(cross-training-domain variant)"
        ),
        "finding_reference": (
            "F#669 (≥9 reuses, promotion confirmed at F#698/F#699); "
            "3rd MEMENTO-cluster child preempt-KILL after F#699 + "
            "exp_memento_block_size_ablation"
        ),
        "parent_experiment": "exp_memento_gemma4_replication",
        "parent_status_at_claim": "provisional",
        "parent_finding": (
            "F#685 (PROVISIONAL design-only, MEMENTO 2-stage SFT not "
            "executable via mlx_lm.lora CLI)"
        ),
        "sibling_precedents": [
            "exp_memento_compression_ratio_benchmark (F#699, 1st MEMENTO-cluster child, single-config)",
            "exp_memento_block_size_ablation (≥9th F#669 reuse, 2nd MEMENTO-cluster child, 1st multi-parent-run sub-axis obs — scalar-sweep)",
        ],
        "all_pass": False,
        "is_smoke": False,
        "kill_criteria": [
            {
                "id": 1906,
                "text": (
                    "MEMENTO-GSM8K compression on MMLU < 85% of "
                    "MEMENTO-MMLU accuracy (cross-training-domain transfer "
                    "fails)"
                ),
                "kind": "target",
                "result": "untested",
                "reason": (
                    "preempt-blocked: the accuracy ratio "
                    "acc(M_GSM8K, MMLU) / acc(M_MMLU, MMLU) is 0/0 — "
                    "unidentifiable. M_GSM8K requires a Gemma-4-MEMENTO "
                    "trained on GSM8K-only segment-and-summarize traces; "
                    "M_MMLU requires a Gemma-4-MEMENTO trained on MMLU-only "
                    "traces. Neither checkpoint exists. MEMENTO paper "
                    "authors released Qwen3/Phi-4/Olmo 3 checkpoints on "
                    "pooled OpenMementos (228K traces) only — not per-corpus "
                    "separated — and no Gemma 4 checkpoint at any mixture. "
                    "Substituting a pooled-trained checkpoint, a base "
                    "Gemma 4 cross-benchmark baseline, or prompt-level "
                    "domain shift for training-corpus separation would be "
                    "antipattern-t (silent objective swap from "
                    "cross-TRAINING-domain transfer to cross-BENCHMARK eval "
                    "or prompting-under-pooled-model)."
                ),
            },
        ],
        "kc_set_gating": (
            "F#666-compliant by vacuous quantification — 1 target K1906, "
            "no proxy to pair. Sparser than F#699/block_size_ablation "
            "(both had proxy+quasi-target pairs) but target-only is "
            "defensible: a compression-ratio or routing proxy adds no "
            "information about behavioral cross-domain transfer."
        ),
        "multi_parent_run_sub_axis": (
            "2nd observation (1st = exp_memento_block_size_ablation, "
            "scalar-hyperparameter-sweep variant; 2nd = this, "
            "categorical cross-training-corpus variant). Underlying "
            "structural property: child KC requires M=N parent _impl "
            "checkpoints when parent's _impl validates only single config. "
            "Here N=2 ({GSM8K-only, MMLU-only}) vs block_size's N=4 "
            "({128,256,512,1024}). Canonical promotion at 3rd obs per "
            "mem-pattern-triple-fire; candidate 3rd instances: "
            "exp_hedgehog_rank_ablation_r4_r8_r16, "
            "exp_jepa_scale_sweep_5m_15m_50m."
        ),
        "unblock_condition": (
            "Parent exp_memento_gemma4_replication reaches status=supported "
            "via exp_memento_gemma4_replication_impl (P=3, already filed) "
            "AND N=2 additional parent _impl runs exist at single-corpus "
            "training mixtures: one on GSM8K-only segment-and-summarize "
            "traces, one on MMLU-only traces. Specifically: K1799 (KV "
            "reduction proxy) AND K1800 (task accuracy drop < 5pp vs base) "
            "AND K1801 (KV-channel ablation target) AND K1802 (throughput "
            "target) all SUPPORTED at each per-corpus _impl. No "
            "KC-augmentation needed at re-claim — K1906 already target per "
            "F#666 trivially. Alternative scope-reduction to single-model "
            "cross-benchmark eval (pooled MEMENTO on GSM8K + MMLU) would "
            "collapse to parent's K1800 subset evaluated on two benchmarks "
            "— antipattern-t risk (substitutes mechanism)."
        ),
        "platform_skills_invoked": [
            "/mlx-dev (noted, not used — no code path)",
            "/fast-mlx (noted, not used — no code path)",
        ],
        "base_model": (
            "mlx-community/gemma-4-e4b-it-4bit (per F#627, not loaded)"
        ),
        "memento_paper": (
            "Kontonis et al. arxiv:2604.09852 (Apr 2026) — Qwen3/Phi-4/"
            "Olmo 3 checkpoints on pooled OpenMementos 228K traces, "
            "no per-corpus separation, no Gemma 4"
        ),
        "impl_follow_up_filed": False,
        "impl_follow_up_rationale": (
            "Preempt-structural KILL does not spawn an _impl companion "
            "(per F#687/F#698/F#699 precedent + reviewer.md §5). Unblock "
            "is parent-external: exp_memento_gemma4_replication_impl "
            "already exists at P=3 from parent's PROVISIONAL filing. "
            "Cross-training-domain KC would require N=2 ADDITIONAL _impl "
            "runs at single-corpus mixtures, not a new companion under "
            "this experiment."
        ),
        "f669_reuse_index": (
            "≥10 (≥9 at block_size_ablation's filing; this is next reuse)"
        ),
        "memento_cluster_child_index": 3,
        "f666_compound_subcase": False,
        "notes": (
            "No MLX code was executed. This is a structural preempt-KILL "
            "per F#669. 3rd MEMENTO-cluster child preempt-KILL after "
            "F#699 and exp_memento_block_size_ablation. Compounding "
            "factor: cross-training-domain KC is strictly stronger than "
            "single-mixture measurement (2nd observation of "
            "multi-parent-run sub-axis; 1st observation was the "
            "scalar-hyperparameter-sweep variant in block_size_ablation). "
            "Notes field claimed 'Does MEMENTO compression learned on "
            "math transfer to general knowledge?' but cross-training-"
            "domain transfer requires two distinct per-corpus trained "
            "checkpoints that do not exist — paper authors released "
            "pooled-mixture checkpoints only, on non-Gemma-4 architectures."
        ),
    }


def main() -> None:
    """Entry point — never raises, always writes results.json."""
    results = build_results()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(
        "[preempt-kill] Wrote "
        f"{out} — verdict=KILLED, reason=preempt F#669 (≥10 reuses) + "
        "multi-parent-run sub-axis (cross-training-domain, 2nd obs)"
    )


if __name__ == "__main__":
    main()
