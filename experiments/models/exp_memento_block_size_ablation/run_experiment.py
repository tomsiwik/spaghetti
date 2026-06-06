"""run_experiment.py — exp_memento_block_size_ablation (PREEMPT-KILL).

This experiment is preempt-killed per Finding #669 (≥9 reuses; promotion
threshold confirmed at F#698 / F#699). No MLX code is written because parent
`exp_memento_gemma4_replication` is PROVISIONAL (F#685) and every KC here
requires a Gemma-4-MEMENTO checkpoint at the swept block size that does not
exist (paper authors released only Qwen3 / Phi-4 / Olmo 3, at a fixed block
size — Gemma 4 is not among them at any block size).

Additionally, a block-size sweep over {128, 256, 512, 1024} requires FOUR
independent MEMENTO training runs of parent's _impl — parent's _impl validates
a single configuration. The sweep is strictly stronger than a single-config
measurement and remains preempt-blocked even under P.status = supported at one
block size.

The KC set IS F#666-compliant (K1904 proxy + K1905 quasi-target), so no
compound F#666 block applies. Clean F#669 reuse on parent-target-unverification.

This scaffold writes a well-formed `results.json` so downstream tooling
(reviewer, analyst, DB `experiment complete`) sees a valid artifact. No code
path raises: the script always produces a non-empty `results.json` that
encodes the preempt-kill verdict and structurally-untestable KCs.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    """Return results dict encoding preempt-KILL.

    No MLX import or call is made. No model is loaded. No measurement runs.
    The verdict is structural: parent target-unverified ⇒ child unidentifiable,
    compounded by hyperparameter-sweep-strictly-stronger-than-single-config.
    """
    return {
        "experiment_id": "exp_memento_block_size_ablation",
        "verdict": "KILLED",
        "kill_reason": "preempt-child-parent-target-unverified + hyperparameter-sweep-strictly-stronger-than-single-config",
        "finding_reference": "F#669 (≥9 reuses, promotion confirmed at F#698/F#699); 2nd MEMENTO-cluster child preempt-KILL after F#699",
        "parent_experiment": "exp_memento_gemma4_replication",
        "parent_status_at_claim": "provisional",
        "parent_finding": "F#685 (PROVISIONAL design-only, MEMENTO 2-stage SFT not executable via mlx_lm.lora CLI)",
        "sibling_precedent": "exp_memento_compression_ratio_benchmark (F#699, 4th F#669 reuse, same parent)",
        "all_pass": False,
        "is_smoke": False,
        "kill_criteria": [
            {
                "id": 1904,
                "text": "Block size < 256 produces compression ratio < 2x (too fine-grained)",
                "kind": "proxy",
                "result": "untested",
                "reason": (
                    "preempt-blocked: no Gemma-4-MEMENTO checkpoint exists at any "
                    "block size, let alone block size 128 or 256. Paper authors "
                    "released Qwen3/Phi-4/Olmo 3 checkpoints at a fixed block size "
                    "(paper default 512) only — Gemma 4 is not among them. Computing "
                    "a 'ratio' absent a compressing model yields 1.0x by identity "
                    "(uncompressed/uncompressed) — antipattern-t if substituted. "
                    "Additionally, thresholds 256 and 2x are behaviorally "
                    "uncalibrated (no evidence 1.9x fails vs 2.1x passes at any "
                    "deployment constraint)."
                ),
            },
            {
                "id": 1905,
                "text": "Block size > 512 produces accuracy < 80% of full-context (too coarse)",
                "kind": "quasi-target",
                "result": "untested",
                "reason": (
                    "preempt-blocked: 'compressed-context' arm requires MEMENTO "
                    "block-mask attention loop with mementos in the KV channel at "
                    "block size 1024 — no such Gemma-4-MEMENTO checkpoint exists "
                    "(paper default is block size 512 only). Substituting 'shorter "
                    "context window' or 'text-level chunking' for 'MEMENTO "
                    "block-mask' would be antipattern-t. 80% threshold is "
                    "uncalibrated (paper reports ~5pp drops at default block size, "
                    "far less than 20%)."
                ),
            },
        ],
        "kc_set_gating": "F#666-compliant (1 proxy K1904 + 1 quasi-target K1905) — no compound F#666 block, analogous to F#699 (not F#698)",
        "hyperparameter_sweep_sub_axis": (
            "New sub-axis observation (1st instance; not yet promotion-eligible): "
            "block-size sweep over {128,256,512,1024} requires 4 independent "
            "parent _impl training runs. Parent's _impl validates a single "
            "configuration (paper default 512). Sweep KC is strictly stronger "
            "than single-config measurement — preempt-blocked even under "
            "P.status=supported at one block size. Tighter unblock condition: "
            "N=4 parent _impl runs, not just 1."
        ),
        "unblock_condition": (
            "Parent exp_memento_gemma4_replication reaches status=supported via "
            "exp_memento_gemma4_replication_impl (P=3, already filed) AND N=4 "
            "additional parent _impl runs exist at block sizes ∈ {128,256,512,1024}. "
            "Specifically: K1799 (KV reduction proxy) AND K1800 (GSM8K target) AND "
            "K1801 (KV-channel ablation target) AND K1802 (throughput target) all "
            "SUPPORTED at each swept block size. No KC-augmentation needed at "
            "re-claim — K1905 already provides quasi-target gate per F#666. "
            "Alternative scope-reduction to single-config measurement would "
            "collapse to parent's K1800 subset (antipattern-t risk)."
        ),
        "platform_skills_invoked": [
            "/mlx-dev (noted, not used — no code path)",
            "/fast-mlx (noted, not used — no code path)",
        ],
        "base_model": "mlx-community/gemma-4-e4b-it-4bit (per F#627, not loaded)",
        "memento_paper": "Kontonis et al. arxiv:2604.09852 (Apr 2026) — Qwen3/Phi-4/Olmo 3 checkpoints at fixed block size, no Gemma 4",
        "impl_follow_up_filed": False,
        "impl_follow_up_rationale": (
            "Preempt-structural KILL does not spawn an _impl companion (per "
            "F#687/F#698/F#699 precedent + reviewer.md §5). Unblock is "
            "parent-external: exp_memento_gemma4_replication_impl already "
            "exists at P=3 from parent's PROVISIONAL filing. Block-size sweep "
            "would require N-1 ADDITIONAL _impl runs, not a new companion "
            "under this experiment."
        ),
        "f669_reuse_index": "≥9 (promotion threshold confirmed at F#698 / F#699)",
        "memento_cluster_child_index": 2,
        "f666_compound_subcase": False,
        "notes": (
            "No MLX code was executed. This is a structural preempt-KILL per "
            "F#669. 2nd MEMENTO-cluster child preempt-KILL after F#699. "
            "Compounding factor: hyperparameter-sweep is strictly stronger "
            "than single-config measurement (new sub-axis candidate, 1st "
            "observation). Notes field claimed 'Sweep the block size parameter "
            "in MEMENTO compression' but block size is a MEMENTO training-time "
            "hyperparameter, not a run-time knob — sweep requires N independent "
            "trained checkpoints that do not exist."
        ),
    }


def main() -> None:
    """Entry point — never raises, always writes results.json."""
    results = build_results()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[preempt-kill] Wrote {out} — verdict=KILLED, reason=preempt F#669 (≥9 reuses) + hyperparameter-sweep sub-axis")


if __name__ == "__main__":
    main()
