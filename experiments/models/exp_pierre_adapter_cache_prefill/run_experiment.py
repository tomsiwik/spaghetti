"""run_experiment.py — exp_pierre_adapter_cache_prefill (PREEMPT-KILL).

This experiment is preempt-killed per Finding #669. No MLX code is written
because parent `exp_prod_mlxlm_integration` is KILLED (F#570, 5 preconditions
T1B/T1C/T2/T3/DEP all fail) and K1913/K1914 require a multi-adapter serving
harness on `pierre-g4e4b` with an idle-time pre-fill scheduler to wall-clock
first-token latency reduction and cache memory overhead. The harness does
not exist:
  - mlx-lm 0.31.2 has no plugin/loader API (T1B).
  - No `pierre-g4e4b` checkpoint on disk or in HF cache (T1C).
  - mlx_lm server body schema validates `adapter` as single `str`, not
    multi-adapter list (T2, verified at mlx_lm/server.py:1155,1236).
  - Trained baseline adapter .safetensors missing for math/code/medical
    domains (T3).
  - Grandparent `exp_prod_pip_package_pierre` KILLED (DEP).

Plus parent-extension requirements (idle-time hook, cache scheduler, TTFT
and memory-delta instrumentation) that are beyond F#570's single-adapter
loader scope.

K1913 (pre-fill doesn't reduce first-token latency by > 20%) and
K1914 (pre-fill memory overhead > 2GB) are both engineering targets —
there is no proxy that could be paired without requiring the same
parent-harness precondition. The KC set is F#666-compliant by vacuous
quantification (no proxy → no pairing obligation). This is the 13th
F#669 reuse overall; it is the 2nd Pierre-serving-cluster F#669 child
(within-cluster reuse after F#740).

The target-only-KC-panel-under-preempt-KILL micro-pattern is already
canonical as of F#740 (cross-cluster triple-fire: F#738 + F#739 + F#740).
This observation is the 4th observation and the 1st within-cluster reuse
inside the Pierre-serving cluster — tally-only, not a canonicalization
event.

Sub-axis classification: single-config idle-time pre-fill (2 engineering
metrics on one configuration). Closest to F#739 (single-config engineering
target-only, MEMENTO) — this is the 2nd observation of that specific
sub-axis variant. Does NOT advance the multi-parent-run sub-axis counter
(remains at 2 obs: F#737 + F#738).

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
    started. No TTFT is timed. No memory is probed. The verdict is
    structural: parent target-unverified; multi-adapter serving harness
    with idle-time pre-fill scheduler does not exist; TTFT reduction
    and cache memory overhead are both unidentifiable (NaN and NaN).
    """
    return {
        "experiment_id": "exp_pierre_adapter_cache_prefill",
        "verdict": "KILLED",
        "kill_reason": (
            "preempt-child-parent-target-unverified "
            "(engineering-target-only KC panel, single-config idle-time "
            "pre-fill; first-token latency reduction and cache memory "
            "overhead require a multi-adapter serving harness on "
            "pierre-g4e4b with an idle-time pre-fill scheduler that does "
            "not exist)"
        ),
        "finding_reference": (
            "F#669 (≥13 reuses; promotion confirmed at F#698/F#699; "
            "target-only-KC-panel canonicalized at F#740 via cross-cluster "
            "triple-fire); 2nd Pierre-serving-cluster F#669 child "
            "(within-cluster reuse after F#740; single-config idle-time "
            "pre-fill sub-axis variant, distinct from F#740's N-spot-"
            "measurement at N∈{3,5}); target-only-KC-panel 4th obs "
            "(post-canonical; 1st within-cluster reuse in Pierre-serving)"
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
            "exp_pierre_multi_adapter_serving_throughput (F#740, 1st Pierre-serving-cluster child, N-spot-measurement at N∈{3,5}, target-only-KC-panel CANONICALIZED cross-cluster triple-fire)",
        ],
        "all_pass": False,
        "is_smoke": False,
        "kill_criteria": [
            {
                "id": 1913,
                "text": (
                    "Pre-fill doesn't reduce first-token latency by > 20%"
                ),
                "kind": "target",
                "result": "untested",
                "reason": (
                    "preempt-blocked: TTFT reduction is undefined — neither "
                    "cold-path TTFT nor warm-path TTFT is measurable without "
                    "a multi-adapter serving harness on pierre-g4e4b with "
                    "an idle-time pre-fill scheduler. mlx-lm 0.31.2 has no "
                    "plugin/loader API (F#570 T1B). No pierre-g4e4b "
                    "checkpoint exists on disk or in HF cache (F#570 T1C). "
                    "mlx_lm server body schema validates 'adapter' as "
                    "single str, not multi-adapter list (F#570 T2, "
                    "mlx_lm/server.py:1155,1236) — single-adapter schema "
                    "makes pre-fill a no-op because the single adapter is "
                    "always resident. Trained adapter safetensors missing "
                    "for math/code/medical domains (F#570 T3) blocks both "
                    "cold and warm path instantiation. Even at F#570 "
                    "SUPPORTED, parent-extension is required: idle-time "
                    "hook in serve-loop, cache scheduler with pre-fill "
                    "policy, TTFT instrumentation distinguishing cold "
                    "(cache miss) from warm (cache hit). Substituting "
                    "single-stack serve with no cache layer, pre-loading "
                    "one adapter at startup, os.fork-COW duplicate as "
                    "'cache', base gemma-4-e4b-it-4bit without Pierre "
                    "wrapper, or torch prefetch into OS page cache would "
                    "each be antipattern-t (silent objective swap) and/or "
                    "antipattern-m (proxy-model-substitution)."
                ),
            },
            {
                "id": 1914,
                "text": (
                    "Pre-fill memory overhead > 2GB"
                ),
                "kind": "target",
                "result": "untested",
                "reason": (
                    "preempt-blocked: cache memory overhead is undefined "
                    "— no cache layer exists, no pierre-g4e4b base to load, "
                    "and no adapter safetensors to pre-fill into a "
                    "hypothetical cache. Memory overhead depends on "
                    "whether adapters are pre-decoded or stored in packed "
                    "safetensor form, on Metal Heap pinning behavior for "
                    "the cache region, and on allocator fragmentation "
                    "interactions with the base-model KV cache — all "
                    "strictly empirical, not derivable from adapter-config "
                    "files. Even at parent SUPPORTED (F#570 resolved), "
                    "an adapter-cache layer with clean memory-attribution "
                    "(cache region vs base model vs KV cache) is a "
                    "parent-extension beyond F#570's single-adapter loader "
                    "scope. Back-deriving memory from os.path.getsize("
                    "adapter.safetensors) would ignore tensor "
                    "deserialization overhead, Metal Heap pinning, and "
                    "allocator fragmentation (antipattern-t)."
                ),
            },
        ],
        "kc_set_gating": (
            "F#666-compliant by vacuous quantification — 2 engineering "
            "targets (K1913, K1914), no proxy to pair. Engineering-target-"
            "only KC panels satisfy F#666 trivially: TTFT-reduction-% and "
            "memory-overhead-GB ARE the targets, no behavioral proxy "
            "exists that doesn't require the same parent-harness "
            "precondition. Post-canonical reuse (pattern canonicalized at "
            "F#740 via cross-cluster triple-fire: F#738 + F#739 + F#740). "
            "This is the 4th observation and 1st within-cluster reuse "
            "inside the Pierre-serving cluster (after F#740); confirms "
            "canonical form's within-cluster portability across sub-axis "
            "variants (F#740 N-spot-measurement vs this single-config "
            "idle-time pre-fill). Tally-only, not a new canonicalization "
            "event."
        ),
        "multi_parent_run_sub_axis": (
            "NOT ADVANCED (reviewer call). This experiment is a single-"
            "config idle-time pre-fill measurement (2 engineering metrics "
            "on one configuration). Structurally closest to F#739 "
            "(single-config engineering target-only, MEMENTO) — this is "
            "the 2nd observation of single-config-target-only-engineering "
            "sub-axis variant. Distinct from F#699 single-config (one N, "
            "proxy+target), F#737 canonical scalar-sweep (one metric "
            "across many N), F#738 categorical cross-corpus, F#740 "
            "serving-config spot-measurement at N∈{3,5}. Does NOT "
            "advance the multi-parent-run sub-axis counter — that counter "
            "tracks same-metric-across-configs sweeps, and this "
            "experiment is single-config (2 distinct metrics, 1 config). "
            "Multi-parent-run sub-axis remains at 2 observations (F#737 "
            "scalar-sweep + F#738 categorical); canonical promotion still "
            "pending a genuine 3rd same-metric-across-configs observation. "
            "Candidate 3rd instances unchanged: exp_hedgehog_rank_ablation_"
            "r4_r8_r16, exp_jepa_scale_sweep_5m_15m_50m, "
            "exp_g4_lora_rank_importance_per_task, "
            "exp_g4_adapter_initialization_comparison."
        ),
        "target_only_kc_panel_micro_pattern": (
            "POST-CANONICAL REUSE (4th obs; canonicalized at F#740). "
            "1st obs: F#738 behavioral target-only (MEMENTO cluster). "
            "2nd obs: F#739 engineering target-only (MEMENTO cluster). "
            "3rd obs: F#740 engineering target-only (Pierre-serving "
            "cluster) — canonicalization event via cross-cluster triple-"
            "fire. 4th obs: this — engineering target-only, Pierre-"
            "serving cluster, within-cluster reuse. Tally only; no new "
            "canonicalization. Strengthens canonical form by confirming "
            "within-cluster portability across distinct sub-axis variants "
            "(F#740 N-spot-measurement + this single-config idle-time "
            "pre-fill)."
        ),
        "single_config_engineering_target_only_variant": (
            "2nd observation of single-config-target-only-engineering "
            "sub-axis variant. 1st obs: F#739 (MEMENTO cluster, realtime "
            "latency). 2nd obs: this (Pierre-serving cluster, adapter-"
            "cache pre-fill). Cross-cluster reuse. Not yet a canonical "
            "variant (would need 3 obs per the triple-fire rule applied "
            "to each sub-axis); 1 more distinct obs would canonicalize "
            "this variant as a stand-alone cluster. Reviewer may flag "
            "for watchlist."
        ),
        "unblock_condition": (
            "Parent exp_prod_mlxlm_integration reaches status=supported "
            "via resolution of all 5 F#570 preconditions AND parent-"
            "extension: (1) T1B — in-tree wrapper or upstream mlx-lm "
            "plugin API (multi-file upstream change). (2) T1C — "
            "pierre-g4e4b composite on disk or HF cache. (3) T2 — body "
            "schema extended from single-adapter str to multi-adapter "
            "list (pre-fill requires multiple adapter identities). (4) "
            "T3 — trained adapter safetensors on disk for math/code/"
            "medical domains. (5) DEP — exp_prod_pip_package_pierre "
            "resolved. PLUS parent-extension: (i) idle-time hook in "
            "serve-loop (on_idle / before_next_request / asyncio-based "
            "scheduler) with well-defined semantics; (ii) adapter-cache "
            "layer with pre-fill policy (next-likely-adapter hypothesis) "
            "and eviction strategy; (iii) TTFT instrumentation "
            "distinguishing cold path (cache miss) from warm path (cache "
            "hit); (iv) memory-overhead probe isolating cache region "
            "from base model and KV cache. No KC-augmentation needed at "
            "re-claim — both KCs already engineering targets per F#666 "
            "trivially. Substituting TTFT-without-cache-layer would "
            "measure baseline cold-path TTFT not pre-fill reduction — "
            "antipattern-t risk."
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
            "T1B_plugin_loader_api": "fail (mlx-lm 0.31.2 has no mlx_lm.loaders/plugins/providers entry-point group); blocks multi-adapter dispatch substrate that pre-fill cache hooks into",
            "T1C_pierre_g4e4b_checkpoint": "fail (0 matches in ~/.cache/huggingface/hub; no experiments/models/pierre-g4e4b/); blocks base-model load for TTFT measurement",
            "T2_multi_adapter_body_schema": "fail (mlx_lm/server.py:1236 validates body['adapter'] as single str); blocks multi-adapter identity set that pre-fill scheduler requires",
            "T3_adapter_safetensors": "fail (exp_p1_t2_single_domain_training/adapters/{math,code,medical}/adapters.safetensors absent); blocks cold-path baseline and warm-path cache-hit measurement",
            "DEP_prod_pip_package_pierre": "fail (KILLED); blocks headline pierre serve --cache-prefill on CLI",
        },
        "parent_extension_requirements": [
            "idle-time hook in serve-loop (on_idle / before_next_request / asyncio-scheduler)",
            "adapter-cache layer with pre-fill policy (next-likely-adapter hypothesis)",
            "cache eviction strategy (LRU / explicit-pin / heuristic)",
            "TTFT instrumentation distinguishing cold path (miss) from warm path (hit)",
            "memory-overhead probe isolating cache region from base model and KV cache",
        ],
        "impl_follow_up_filed": False,
        "impl_follow_up_rationale": (
            "Preempt-structural KILL does not spawn an _impl companion "
            "(per F#687/F#698/F#699/F#737/F#738/F#739/F#740 precedent + "
            "reviewer.md §5). Unblock is parent-external (F#570 "
            "resolution) plus parent-extension (idle-time hook + cache "
            "scheduler + TTFT and memory-delta instrumentation); both "
            "are at the Pierre-serving-infrastructure layer, not under "
            "this child. F#570's own _impl would resolve preconditions "
            "T1B/T2 (schema/API) but the parent-extension requirements "
            "(idle-time hook, cache scheduler, TTFT instrumentation) "
            "remain — same structural pattern as F#740."
        ),
        "f669_reuse_index": (
            "13th (12th was F#740 exp_pierre_multi_adapter_serving_"
            "throughput); 2nd Pierre-serving-cluster F#669 child (1st "
            "within-cluster reuse after F#740)"
        ),
        "cluster_child_index": {
            "cluster": "Pierre-serving (parent F#570)",
            "child_index_within_cluster": 2,
            "prior_pierre_serving_f669_children": [
                "F#740 (exp_pierre_multi_adapter_serving_throughput, 2-point serving-config spot-measurement at N∈{3,5})",
            ],
            "prior_f652_non_f669_children": [
                "F#655 (ap-017 §s4 T5-K via F#652, software-infra-unbuilt route, not F#669)",
                "F#657 (ap-017 28th composition-bug via F#652, not F#669)",
            ],
        },
        "f666_compound_subcase": False,
        "kc_kind_composition": "target+target (engineering)",
        "kc_panel_variant_classification": (
            "target-only engineering (2×); 4th observation of target-"
            "only-KC-panel-under-preempt-KILL micro-pattern; post-"
            "canonical (pattern canonicalized at F#740). 1st within-"
            "cluster reuse inside Pierre-serving cluster. Sub-axis: "
            "single-config idle-time pre-fill (2nd obs of single-config-"
            "target-only-engineering variant after F#739)."
        ),
        "consolidation_candidate": (
            "Pierre-serving cluster now has 2 F#669 children (F#740 + "
            "this); remaining P≤2 open serving/p1 experiments are likely "
            "to preempt-KILL identically. Reviewer may elect to "
            "consolidate future Pierre-serving preempt-KILLs into a "
            "single cluster-level finding rather than filing one F#669-"
            "reuse finding per child. This experiment still files its "
            "own F#669 13th-reuse finding; consolidation is a recommended "
            "option for subsequent Pierre-serving children."
        ),
        "notes": (
            "No MLX code was executed. This is a structural preempt-KILL "
            "per F#669. 2nd Pierre-serving-cluster F#669 child preempt-"
            "KILL (parent F#570); F#669 reuse count advances to 13th "
            "overall. Target-only-KC-panel-under-preempt-KILL micro-"
            "pattern is post-canonical (canonicalized at F#740 via "
            "cross-cluster triple-fire). This observation is the 4th "
            "target-only-panel obs and the 1st within-cluster reuse "
            "inside Pierre-serving cluster — tally only, confirms "
            "canonical form's within-cluster portability across sub-"
            "axis variants. Sub-axis classification: single-config "
            "idle-time pre-fill — 2nd obs of single-config-target-only-"
            "engineering variant (after F#739). Does NOT advance the "
            "multi-parent-run sub-axis counter. Experiment record has "
            "empty success_criteria (F#702 hygiene issue, noted not "
            "patched this iteration since preempt-KILL supersedes "
            "hygiene correction). Platform field was ~ (null); preempt-"
            "KILL verdict does not depend on platform field since no "
            "code executes. Consolidation of remaining Pierre-serving "
            "children into a single cluster-level finding is a "
            "recommended reviewer option."
        ),
    }


def main() -> None:
    """Entry point — never raises, always writes results.json."""
    results = build_results()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(
        "[preempt-kill] Wrote "
        f"{out} — verdict=KILLED, reason=preempt F#669 (13th reuse), "
        "2nd Pierre-serving-cluster child, target-only-panel POST-"
        "CANONICAL reuse (4th obs; 1st within-cluster reuse in Pierre-"
        "serving)"
    )


if __name__ == "__main__":
    main()
