"""run_experiment.py — exp_memento_compression_ratio_benchmark (PREEMPT-KILL).

This experiment is preempt-killed per Finding #669 (4th reuse; promotion
threshold confirmed at F#698 3rd reuse). No MLX code is written because parent
`exp_memento_gemma4_replication` is PROVISIONAL (F#685) and every KC here
requires a Gemma-4-MEMENTO checkpoint that does not exist (paper authors
released only Qwen3 / Phi-4 / Olmo 3 — Gemma 4 is not among them).

Unlike F#698, the KC set IS properly target-gated (K1850 proxy + K1851 target
per F#666), so no compound block applies. Single preempt-block on parent
target-unverification only.

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
    The verdict is structural: parent target-unverified ⇒ child unidentifiable.
    """
    return {
        "experiment_id": "exp_memento_compression_ratio_benchmark",
        "verdict": "KILLED",
        "kill_reason": "preempt-child-parent-target-unverified",
        "finding_reference": "F#669 (4th reuse; promotion threshold per F#698 already confirmed)",
        "parent_experiment": "exp_memento_gemma4_replication",
        "parent_status_at_claim": "provisional",
        "parent_finding": "F#685 (PROVISIONAL design-only, MEMENTO 2-stage SFT not executable via mlx_lm.lora CLI)",
        "all_pass": False,
        "is_smoke": False,
        "kill_criteria": [
            {
                "id": 1850,
                "text": "MEMENTO compression ratio < 3x (not worth the SFT cost)",
                "kind": "proxy",
                "result": "untested",
                "reason": (
                    "preempt-blocked: no Gemma-4-MEMENTO checkpoint exists. "
                    "Compression ratio is undefined absent a model that performs "
                    "compression. Paper authors released Qwen3/Phi-4/Olmo 3 "
                    "checkpoints only — not Gemma 4. Vacuous 1.0x by identity if "
                    "computed against base Gemma 4 (uncompressed/uncompressed)."
                ),
            },
            {
                "id": 1851,
                "text": "Compressed-context accuracy < 85% of full-context on GSM8K",
                "kind": "target",
                "result": "untested",
                "reason": (
                    "preempt-blocked: 'compressed-context' arm requires MEMENTO "
                    "block-mask attention loop with mementos in the KV channel — "
                    "the precise mechanism parent F#685 has not yet validated. "
                    "Substituting 'shorter context window' for 'MEMENTO compression' "
                    "would be antipattern-t (silent objective swap)."
                ),
            },
        ],
        "kc_set_gating": "F#666-compliant (1 proxy K1850 + 1 target K1851) — no compound F#666 block, unlike F#698",
        "unblock_condition": (
            "Parent exp_memento_gemma4_replication reaches status=supported via "
            "exp_memento_gemma4_replication_impl (P=3, already filed). Specifically: "
            "K1799 (KV reduction proxy) AND K1800 (GSM8K target) AND K1801 "
            "(KV-channel ablation target) AND K1802 (throughput target) all SUPPORTED. "
            "No KC-augmentation needed at re-claim — K1851 already provides target gate."
        ),
        "platform_skills_invoked": [
            "/mlx-dev (noted, not used — no code path)",
            "/fast-mlx (noted, not used — no code path)",
        ],
        "base_model": "mlx-community/gemma-4-e4b-it-4bit (per F#627, not loaded)",
        "memento_paper": "Kontonis et al. arxiv:2604.09852 (Apr 2026) — Qwen3/Phi-4/Olmo 3 checkpoints, no Gemma 4",
        "impl_follow_up_filed": False,
        "impl_follow_up_rationale": (
            "Preempt-structural KILL does not spawn an _impl companion (per F#687/F#698 "
            "precedent + reviewer.md §5). Unblock is parent-external: "
            "exp_memento_gemma4_replication_impl already exists at P=3 from parent's "
            "PROVISIONAL filing."
        ),
        "f669_reuse_index": 4,
        "f666_compound_subcase": False,
        "notes": (
            "No MLX code was executed. This is a structural preempt-KILL per F#669. "
            "Fourth reuse of the preempt-child-parent-target-unverified pattern "
            "(F#669 → F#687 → F#698 → this). 4th reuse re-confirms the promotion "
            "threshold reached at F#698 (3rd reuse). Notes field on the experiment "
            "claimed 'No dependency on full replication' but this is materially false: "
            "no Gemma-4-MEMENTO checkpoint exists publicly."
        ),
    }


def main() -> None:
    """Entry point — never raises, always writes results.json."""
    results = build_results()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[preempt-kill] Wrote {out} — verdict=KILLED, reason=preempt F#669 (4th reuse)")


if __name__ == "__main__":
    main()
