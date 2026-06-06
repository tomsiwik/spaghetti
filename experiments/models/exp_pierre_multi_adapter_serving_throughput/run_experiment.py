"""run_experiment.py — exp_pierre_multi_adapter_serving_throughput (PREEMPT-KILL).

This experiment is preempt-killed per Finding #669. No MLX code is written
because parent `exp_prod_mlxlm_integration` is KILLED (F#570, 5 preconditions
T1B/T1C/T2/T3/DEP all fail) and K1911/K1912 require a multi-adapter concurrent
serving harness on `pierre-g4e4b` to wall-clock throughput ratio and peak
memory. The harness does not exist:
  - mlx-lm 0.31.2 has no plugin/loader API (T1B).
  - No `pierre-g4e4b` checkpoint on disk or in HF cache (T1C).
  - mlx_lm server body schema validates `adapter` as single `str`, not
    multi-adapter list (T2, verified at mlx_lm/server.py:1155,1236).
  - Trained baseline adapter .safetensors missing for math/code/medical
    domains (T3).
  - Grandparent `exp_prod_pip_package_pierre` KILLED (DEP).

K1911 (N=3 concurrent-stack throughput < 50% of single-stack) and
K1912 (memory > 40GB at N=5 stacks) are both engineering targets —
there is no proxy that could be paired without requiring the same
parent-harness precondition. The KC set is F#666-compliant by vacuous
quantification (no proxy → no pairing obligation). This is the 12th
F#669 reuse overall; it is the 1st Pierre-serving-cluster child
preempt-KILL (new cluster — parent F#570, distinct from the MEMENTO
cluster which contained F#699/F#737/F#738/F#739).

The target-only-KC-panel-under-preempt-KILL micro-pattern canonicalizes
at this observation: 3rd obs (1st F#738 behavioral / MEMENTO; 2nd F#739
engineering / MEMENTO; 3rd this engineering / Pierre-serving). Cross-
cluster triple-fire promotes watchlist → canonical.

Sub-axis classification: 2-point serving-config spot-measurement at
N∈{3,5} with implicit N=1 baseline. Distinct from F#699 single-config
(one N), F#737 scalar-sweep (one metric across many N), and F#738
categorical cross-corpus. Conservatively does NOT advance the multi-
parent-run sub-axis counter — reviewer call per mem-pattern-triple-fire.

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

    No MLX import or call is made. No model is loaded. No server is
    started. No throughput is timed. No memory is probed. The verdict
    is structural: parent target-unverified; multi-adapter concurrent
    serving harness does not exist; throughput ratio and peak memory
    are both unidentifiable (NaN/NaN and NaN respectively).
    """
    return {
        "experiment_id": "exp_pierre_multi_adapter_serving_throughput",
        "verdict": "KILLED",
        "kill_reason": (
            "preempt-child-parent-target-unverified "
            "(engineering-target-only KC panel, 2-point serving-config "
            "spot-measurement at N∈{3,5}; N=3 concurrent-stack throughput "
            "ratio and N=5 peak memory require a multi-adapter concurrent "
            "serving harness on pierre-g4e4b that does not exist)"
        ),
        "finding_reference": (
            "F#669 (≥12 reuses; promotion confirmed at F#698/F#699); "
            "1st Pierre-serving-cluster F#669 child (new cluster — parent "
            "F#570, distinct from MEMENTO cluster F#699/F#737/F#738/F#739); "
            "target-only-KC-panel-under-preempt-KILL 3rd obs cross-cluster "
            "triple-fire (promotes watchlist → canonical)"
        ),
        "parent_experiment": "exp_prod_mlxlm_integration",
        "parent_status_at_claim": "killed",
        "parent_finding": (
            "F#570 (KILLED: 5 preconditions T1B/T1C/T2/T3/DEP all fail; "
            "mlx-lm 0.31.2 has no plugin/loader API; no pierre-g4e4b "
            "checkpoint; server body schema single-adapter str; trained "
            "adapter safetensors missing; exp_prod_pip_package_pierre "
            "KILLED)"
        ),
        "grandparent_experiment": "exp_prod_pip_package_pierre",
        "grandparent_status_at_claim": "killed",
        "sibling_precedents": [
            "exp_memento_compression_ratio_benchmark (F#699, 1st MEMENTO-cluster child, single-config, proxy+target)",
            "exp_memento_block_size_ablation (F#737, 2nd MEMENTO-cluster child, scalar-sweep, multi-parent-run sub-axis 1st obs)",
            "exp_memento_cross_domain_transfer (F#738, 3rd MEMENTO-cluster child, categorical cross-corpus, target-only-behavioral 1st obs)",
            "exp_memento_realtime_latency (F#739, 4th MEMENTO-cluster child, single-config, target-only-engineering 2nd obs)",
        ],
        "all_pass": False,
        "is_smoke": False,
        "kill_criteria": [
            {
                "id": 1911,
                "text": (
                    "N=3 concurrent adapter stacks throughput < 50% of "
                    "single-stack"
                ),
                "kind": "target",
                "result": "untested",
                "reason": (
                    "preempt-blocked: throughput ratio is NaN/NaN — neither "
                    "N=3 concurrent-stack throughput nor N=1 single-stack "
                    "baseline is measurable without a multi-adapter serving "
                    "harness on pierre-g4e4b. mlx-lm 0.31.2 has no plugin/"
                    "loader API (F#570 T1B). No pierre-g4e4b checkpoint "
                    "exists on disk or in HF cache (F#570 T1C). mlx_lm "
                    "server body schema validates 'adapter' as single str, "
                    "not multi-adapter list (F#570 T2, mlx_lm/server.py:"
                    "1155,1236). Trained adapter safetensors missing for "
                    "math/code/medical domains (F#570 T3). Substituting "
                    "single-stack repeated serving (no concurrency), "
                    "sequential N=1 timings summed as 'N=3 concurrent', "
                    "base gemma-4-e4b-it-4bit without Pierre wrapper, or "
                    "vLLM/llama.cpp cross-framework numbers would each be "
                    "antipattern-t (silent objective swap) and/or "
                    "antipattern-m (proxy-model-substitution)."
                ),
            },
            {
                "id": 1912,
                "text": (
                    "Memory usage > 40GB at N=5 stacks (exceeds M5 Pro 48GB)"
                ),
                "kind": "target",
                "result": "untested",
                "reason": (
                    "preempt-blocked: peak memory at N=5 stacks is NaN — no "
                    "N=5 multi-adapter loader exists and no pierre-g4e4b "
                    "base to load into memory. Memory at N=5 depends on "
                    "whether adapters are resident-all (≈5× baseline) or "
                    "LRU-paged (≤2× baseline), on the unified-memory "
                    "allocator's fragmentation behavior, and on KV-cache "
                    "footprint scaling with concurrency — all strictly "
                    "empirical, not derivable from adapter-config files. "
                    "Even at parent SUPPORTED (F#570 resolved), N≥5 "
                    "resident-stack loading and RSS + Metal-unified-pool "
                    "instrumentation are parent-extensions beyond F#570's "
                    "single-adapter-serving scope. Back-deriving memory "
                    "from analytical 5×adapter + base sum would ignore "
                    "allocator fragmentation and Metal Heap pinning "
                    "(antipattern-t)."
                ),
            },
        ],
        "kc_set_gating": (
            "F#666-compliant by vacuous quantification — 2 engineering "
            "targets (K1911, K1912), no proxy to pair. Engineering-target-"
            "only KC panels satisfy F#666 trivially: throughput-ratio and "
            "peak-memory ARE the targets, no behavioral proxy exists that "
            "doesn't require the same parent-harness precondition. 3rd "
            "observation of target-only-KC-panel-under-preempt-KILL "
            "micro-pattern after F#738 (behavioral, MEMENTO) and F#739 "
            "(engineering, MEMENTO); this is engineering target-only in "
            "the Pierre-serving cluster — CROSS-CLUSTER TRIPLE-FIRE. "
            "Promotes watchlist → canonical per mem-pattern-triple-fire."
        ),
        "multi_parent_run_sub_axis": (
            "AMBIGUOUS (reviewer call). This experiment is a 2-point "
            "serving-config spot-measurement at N∈{3,5} (plus implicit N=1 "
            "baseline for K1911's ratio), distinct from: F#699 single-"
            "config (one N), F#737 canonical scalar-sweep (one metric "
            "across many N), F#738 categorical cross-corpus. Conservatively "
            "classified as 'serving-config spot-measurement (target-only "
            "engineering)' — its own variant, not a standard scalar-sweep, "
            "because K1911/K1912 measure distinct metrics at different N "
            "points rather than the same metric across a range. Does NOT "
            "automatically advance the multi-parent-run sub-axis counter. "
            "Multi-parent-run sub-axis remains at 2 observations (F#737 "
            "scalar-sweep + F#738 categorical); canonical promotion still "
            "pending a genuine 3rd same-metric-across-configs observation. "
            "Candidate 3rd instances unchanged: exp_hedgehog_rank_ablation_"
            "r4_r8_r16, exp_jepa_scale_sweep_5m_15m_50m, "
            "exp_g4_lora_rank_importance_per_task, "
            "exp_g4_adapter_initialization_comparison."
        ),
        "target_only_kc_panel_micro_pattern": (
            "CANONICALIZED at this observation (3rd obs, cross-cluster "
            "triple-fire). 1st obs: F#738 behavioral target-only (MEMENTO "
            "cluster). 2nd obs: F#739 engineering target-only (MEMENTO "
            "cluster). 3rd obs: this — engineering target-only, Pierre-"
            "serving cluster. Cross-cluster independence strengthens the "
            "pattern beyond within-cluster triple-fire: the micro-pattern "
            "is not confined to a single parent's idiosyncrasies. "
            "Engineering + behavioral targets both admit F#666-compliance "
            "via vacuous quantification because neither requires a pairable "
            "proxy (proxies would, in every observed case, require the "
            "same parent-impl precondition and add no identifying power)."
        ),
        "unblock_condition": (
            "Parent exp_prod_mlxlm_integration reaches status=supported "
            "via resolution of all 5 F#570 preconditions AND parent-"
            "extension: (1) T1B — in-tree wrapper or upstream mlx-lm "
            "plugin API (multi-file upstream change). (2) T1C — "
            "pierre-g4e4b composite on disk or HF cache. (3) T2 — body "
            "schema extended from single-adapter str to multi-adapter "
            "list. (4) T3 — trained adapter safetensors on disk for "
            "math/code/medical domains. (5) DEP — exp_prod_pip_package_"
            "pierre resolved. PLUS parent-extension: (i) concurrency "
            "scheduler holds N≥5 resident adapter stacks in unified "
            "memory (or validated LRU-paging policy); (ii) RSS + Metal-"
            "unified-pool peak instrumentation under active serving "
            "workload; (iii) ≥3 concurrent-client driver with realistic "
            "prompt distribution. No KC-augmentation needed at re-claim "
            "— both KCs already engineering targets per F#666 trivially. "
            "Substituting serial N=1-then-N=3-then-N=5 timing (no "
            "concurrency) would measure stack-switching cost not "
            "concurrent-serving cost — antipattern-t risk."
        ),
        "platform_skills_invoked": [
            "/mlx-dev (noted, not used — no code path)",
            "/fast-mlx (noted, not used — no code path)",
        ],
        "base_model": (
            "mlx-community/gemma-4-e4b-it-4bit (per F#627; Pierre-stack "
            "base, not loaded). Composite pierre-g4e4b does not exist on "
            "disk or in HF cache (F#570 T1C)."
        ),
        "f570_preconditions_mapped": {
            "T1B_plugin_loader_api": "fail (mlx-lm 0.31.2 has no mlx_lm.loaders/plugins/providers entry-point group); blocks multi-adapter dispatch substrate",
            "T1C_pierre_g4e4b_checkpoint": "fail (0 matches in ~/.cache/huggingface/hub; no experiments/models/pierre-g4e4b/); blocks base-model load",
            "T2_multi_adapter_body_schema": "fail (mlx_lm/server.py:1236 validates body['adapter'] as single str); blocks per-request multi-adapter selection",
            "T3_adapter_safetensors": "fail (exp_p1_t2_single_domain_training/adapters/{math,code,medical}/adapters.safetensors absent); blocks N=1 baseline and N≥3 resident set",
            "DEP_prod_pip_package_pierre": "fail (KILLED); blocks headline pip install pierre CLI",
        },
        "impl_follow_up_filed": False,
        "impl_follow_up_rationale": (
            "Preempt-structural KILL does not spawn an _impl companion "
            "(per F#687/F#698/F#699/F#737/F#738/F#739 precedent + "
            "reviewer.md §5). Unblock is parent-external (F#570 "
            "resolution) plus parent-extension (multi-adapter scheduler + "
            "memory instrumentation); both are at the Pierre-serving-"
            "infrastructure layer, not under this child. F#570's own "
            "_impl would resolve preconditions T1B/T2 (schema/API) but "
            "the parent-extension requirements (concurrency scheduler, "
            "N≥5 residency, memory instrumentation) remain."
        ),
        "f669_reuse_index": (
            "12th (11th was F#739 exp_memento_realtime_latency); 1st "
            "Pierre-serving-cluster F#669 child (new cluster beyond MEMENTO)"
        ),
        "cluster_child_index": {
            "cluster": "Pierre-serving (parent F#570)",
            "child_index_within_cluster": 1,
            "prior_f652_non_f669_children": [
                "F#655 (ap-017 §s4 T5-K via F#652, software-infra-unbuilt route, not F#669)",
                "F#657 (ap-017 28th composition-bug via F#652, not F#669)",
            ],
        },
        "f666_compound_subcase": False,
        "kc_kind_composition": "target+target (engineering)",
        "kc_panel_variant_classification": (
            "target-only engineering (2×); 3rd observation of target-only-"
            "KC-panel-under-preempt-KILL micro-pattern, cross-cluster "
            "triple-fire, CANONICALIZED at this observation."
        ),
        "notes": (
            "No MLX code was executed. This is a structural preempt-KILL "
            "per F#669. 1st Pierre-serving-cluster F#669 child preempt-KILL "
            "(parent F#570); F#669 reuse count advances to 12th overall. "
            "Target-only-KC-panel-under-preempt-KILL micro-pattern "
            "CANONICALIZES (3rd obs, cross-cluster triple-fire: F#738 "
            "behavioral/MEMENTO + F#739 engineering/MEMENTO + this "
            "engineering/Pierre-serving). Sub-axis classification is 2-"
            "point serving-config spot-measurement (N∈{3,5}) — distinct "
            "variant, does NOT automatically advance canonical multi-"
            "parent-run sub-axis counter. Experiment record has empty "
            "success_criteria (F#702 hygiene issue, noted not patched "
            "this iteration since preempt-KILL supersedes hygiene "
            "correction). Platform field was ~ (null); preempt-KILL "
            "verdict does not depend on platform field since no code "
            "executes."
        ),
    }


def main() -> None:
    """Entry point — never raises, always writes results.json."""
    results = build_results()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(
        "[preempt-kill] Wrote "
        f"{out} — verdict=KILLED, reason=preempt F#669 (12th reuse), "
        "1st Pierre-serving-cluster child, target-only-panel CANONICAL "
        "(cross-cluster triple-fire)"
    )


if __name__ == "__main__":
    main()
