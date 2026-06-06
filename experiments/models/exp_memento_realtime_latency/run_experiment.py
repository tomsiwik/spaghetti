"""run_experiment.py — exp_memento_realtime_latency (PREEMPT-KILL).

This experiment is preempt-killed per Finding #669. No MLX code is written
because parent `exp_memento_gemma4_replication` is PROVISIONAL (F#685) and
K1907/K1908 require a callable Gemma-4-MEMENTO block-compression forward
pass to wall-clock per-block and streaming-vs-batch latency. The mechanism
does not exist:
  - No public Gemma-4-MEMENTO checkpoint at any training mixture (paper
    released Qwen3 / Phi-4 / Olmo 3 only, on pooled OpenMementos 228K).
  - MEMENTO 2-stage SFT + block-mask attention is not executable via
    mlx_lm.lora CLI (parent F#685 finding); even with a checkpoint, the
    runtime path needs custom MLX implementation that does not exist.

K1907 (per-block latency > 50ms) and K1908 (streaming > 2x batch latency)
are both engineering targets — there is no proxy that could be paired
without requiring the same parent-impl checkpoint. The KC set is
F#666-compliant by vacuous quantification (no proxy → no pairing
obligation). This is the 4th MEMENTO-cluster child preempt-KILL after
F#699 (compression_ratio_benchmark, single-config) + F#737
(block_size_ablation, scalar-sweep) + F#738 (cross_domain_transfer,
categorical cross-corpus). Single-config engineering-target-only variant;
no new sub-axis observation (multi-parent-run sub-axis remains at 2 obs).

This scaffold writes a well-formed `results.json` so downstream tooling
(reviewer, analyst, DB `experiment complete`) sees a valid artifact. No
code path raises: the script always produces a non-empty `results.json`
that encodes the preempt-kill verdict and structurally-untestable KCs.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    """Return results dict encoding preempt-KILL.

    No MLX import or call is made. No model is loaded. No latency is timed.
    The verdict is structural: parent target-unverified; per-block and
    streaming-vs-batch latency cannot be measured without a callable
    Gemma-4-MEMENTO forward pass that does not exist.
    """
    return {
        "experiment_id": "exp_memento_realtime_latency",
        "verdict": "KILLED",
        "kill_reason": (
            "preempt-child-parent-target-unverified "
            "(engineering-target-only KC panel, single-config; "
            "per-block and streaming-vs-batch latency require a callable "
            "Gemma-4-MEMENTO forward pass that does not exist)"
        ),
        "finding_reference": (
            "F#669 (≥10 reuses, promotion confirmed at F#698/F#699); "
            "4th MEMENTO-cluster child preempt-KILL after F#699 + F#737 + F#738"
        ),
        "parent_experiment": "exp_memento_gemma4_replication",
        "parent_status_at_claim": "provisional",
        "parent_finding": (
            "F#685 (PROVISIONAL design-only, MEMENTO 2-stage SFT + "
            "block-mask attention not executable via mlx_lm.lora CLI)"
        ),
        "sibling_precedents": [
            "exp_memento_compression_ratio_benchmark (F#699, 1st MEMENTO-cluster child, single-config, proxy+target)",
            "exp_memento_block_size_ablation (F#737, 2nd MEMENTO-cluster child, scalar-sweep, multi-parent-run sub-axis 1st obs)",
            "exp_memento_cross_domain_transfer (F#738, 3rd MEMENTO-cluster child, categorical cross-corpus, multi-parent-run sub-axis 2nd obs)",
        ],
        "all_pass": False,
        "is_smoke": False,
        "kill_criteria": [
            {
                "id": 1907,
                "text": (
                    "Per-block compression latency > 50ms on M5 Pro "
                    "(NOT real-time)"
                ),
                "kind": "target",
                "result": "untested",
                "reason": (
                    "preempt-blocked: per-block latency is undefined — no "
                    "callable Gemma-4-MEMENTO block-compression kernel "
                    "exists on this platform. Latency at 50ms granularity "
                    "is dominated by Metal kernel selection, MLX lazy-eval "
                    "boundaries, KV-cache layout in unified memory, and "
                    "tile-size compile decisions — strictly empirical, not "
                    "derivable from architecture alone. Substituting base "
                    "Gemma 4 inference latency (no MEMENTO compression), "
                    "Qwen3/Phi-4/Olmo 3 paper-checkpoint timings as Gemma 4 "
                    "stand-in, hand-written block-mask MLX reimplementation "
                    "without trained weights, or paper Table 4 throughput / "
                    "block-count back-derivation would each be antipattern-t "
                    "(silent objective swap) and/or antipattern-m "
                    "(proxy-model-substitution)."
                ),
            },
            {
                "id": 1908,
                "text": (
                    "Streaming (incremental) compression latency > 2x batch "
                    "compression latency"
                ),
                "kind": "target",
                "result": "untested",
                "reason": (
                    "preempt-blocked: ratio is NaN/NaN — neither streaming "
                    "nor batch compression latency is measurable without a "
                    "callable Gemma-4-MEMENTO forward pass under both "
                    "scheduling regimes. Parent's K1802 (throughput target) "
                    "measures batch tokens/sec only; it does not exercise "
                    "incremental block-by-block compression and does not "
                    "report per-block wall-clock at the granularity K1908 "
                    "requires. Even at parent SUPPORTED, parent-side "
                    "instrumentation (latency hook + streaming-mode forward "
                    "path) would need to be added to surface the "
                    "measurement surface."
                ),
            },
        ],
        "kc_set_gating": (
            "F#666-compliant by vacuous quantification — 2 engineering "
            "targets (K1907, K1908), no proxy to pair. Engineering-target-"
            "only KC panels satisfy F#666 trivially: latency IS the target, "
            "no behavioral proxy exists that doesn't require the same "
            "parent-impl checkpoint. 2nd MEMENTO-cluster child with "
            "target-only KC panel after F#738 (cross_domain_transfer "
            "behavioral target-only); both observations together form a "
            "watchlist micro-pattern (engineering vs behavioral) below the "
            "triple-fire canonicalization threshold."
        ),
        "multi_parent_run_sub_axis": (
            "NOT advanced. Single-config engineering measurement (default "
            "block size 512, default pooled-OpenMementos training) — same "
            "structural class as F#699 (compression_ratio_benchmark). "
            "Multi-parent-run sub-axis remains at 2 observations: 1st = "
            "block_size_ablation (scalar-sweep), 2nd = cross_domain_transfer "
            "(categorical cross-corpus). Canonical promotion still pending "
            "3rd observation per mem-pattern-triple-fire. Candidate 3rd "
            "instances: exp_hedgehog_rank_ablation_r4_r8_r16, "
            "exp_jepa_scale_sweep_5m_15m_50m, "
            "exp_g4_lora_rank_importance_per_task, "
            "exp_g4_adapter_initialization_comparison."
        ),
        "unblock_condition": (
            "Parent exp_memento_gemma4_replication reaches status=supported "
            "via exp_memento_gemma4_replication_impl (P=3, already filed) "
            "AND parent's _impl exposes a per-block latency hook + "
            "streaming-mode forward path. Specifically: K1799 (KV reduction "
            "proxy) AND K1800 (task accuracy drop < 5pp vs base) AND K1801 "
            "(KV-channel ablation target) AND K1802 (throughput target) all "
            "SUPPORTED at parent's _impl PLUS a parent-side engineering "
            "tightening to surface per-block timing and a streaming-mode "
            "flag (latency-instrumented forward pass). Parent's K1802 "
            "measures batch tokens/sec, not per-block wall-clock or "
            "streaming-vs-batch ratio — the measurement surface K1907/K1908 "
            "require is a parent-extension, not a separate _impl companion. "
            "No KC-augmentation needed at re-claim — both KCs already "
            "engineering targets per F#666 trivially. Alternative scope-"
            "reduction to base-Gemma-4 inference latency would substitute "
            "mechanism (base-model decoding ≠ MEMENTO per-block compression) "
            "— antipattern-t risk."
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
            "Olmo 3 checkpoints on pooled OpenMementos 228K traces, no "
            "Gemma 4 at any mixture. Paper Table 4 reports throughput "
            "comparisons but not per-block latency at 50ms-real-time "
            "granularity"
        ),
        "impl_follow_up_filed": False,
        "impl_follow_up_rationale": (
            "Preempt-structural KILL does not spawn an _impl companion "
            "(per F#687/F#698/F#699/F#737/F#738 precedent + reviewer.md §5). "
            "Unblock is parent-external: exp_memento_gemma4_replication_impl "
            "already exists at P=3 from parent's PROVISIONAL filing. K1907/"
            "K1908 additionally require a parent-side engineering tightening "
            "(per-block latency hook + streaming-mode forward path) — this "
            "is a parent-extension, not a new child _impl."
        ),
        "f669_reuse_index": (
            "≥11 (≥10 at cross_domain_transfer's filing; this is next reuse)"
        ),
        "memento_cluster_child_index": 4,
        "f666_compound_subcase": False,
        "kc_kind_composition": "target+target (engineering)",
        "engineering_target_only_micro_pattern_obs": (
            "2nd observation (1st = exp_memento_cross_domain_transfer "
            "behavioral target-only, F#738; 2nd = this engineering target-"
            "only). Watchlist; 3rd obs would canonicalize per mem-pattern-"
            "triple-fire. Below threshold for promotion."
        ),
        "notes": (
            "No MLX code was executed. This is a structural preempt-KILL "
            "per F#669. 4th MEMENTO-cluster child preempt-KILL after F#699 "
            "+ F#737 + F#738. Single-config measurement of engineering "
            "(latency) properties — same structural class as F#699 "
            "(compression_ratio_benchmark, single-config), distinct from "
            "F#737/F#738 (sweep variants, multi-parent-run sub-axis). "
            "Multi-parent-run sub-axis NOT advanced (remains at 2 obs). "
            "Notes field claimed 'measure per-block overhead in MLX' but "
            "per-block compression overhead requires a Gemma-4-MEMENTO "
            "forward pass that does not exist — paper authors released "
            "pooled-mixture checkpoints only, on non-Gemma-4 architectures, "
            "and MEMENTO block-mask attention is not executable via "
            "mlx_lm.lora CLI per parent F#685."
        ),
    }


def main() -> None:
    """Entry point — never raises, always writes results.json."""
    results = build_results()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(
        "[preempt-kill] Wrote "
        f"{out} — verdict=KILLED, reason=preempt F#669 (≥11 reuses), "
        "4th MEMENTO-cluster child (single-config engineering target-only)"
    )


if __name__ == "__main__":
    main()
