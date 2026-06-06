# PAPER: Pierre multi-adapter serving — throughput at N=3 and peak memory at N=5

**Status:** KILLED — preempt-structural (F#669 12th reuse)
**Verdict line:** KILLED (no kill criterion measurable; parent `exp_prod_mlxlm_integration` is KILLED per F#570 with 5 preconditions unresolved; **not** PROVISIONAL — proven unmeasurable on current state)

## TL;DR

Two target-only engineering kill criteria (K1911 N=3 concurrent-stack throughput ratio < 50%; K1912 N=5 peak memory > 40GB on 48GB M5 Pro) both FAIL as **unmeasured** due to parent-target-unverified preempt block. mlx-lm 0.31.2 has no plugin/loader API (F#570 T1B), no `pierre-g4e4b` checkpoint exists (T1C), server body schema validates `adapter` as single `str` not multi-adapter list (T2), trained adapter safetensors missing (T3), and grandparent `exp_prod_pip_package_pierre` is KILLED (DEP). Concurrent multi-adapter serving harness required by both KCs does not exist on this platform.

This is the **12th F#669 reuse overall** and the **1st Pierre-serving-cluster F#669 child preempt-KILL** (new cluster — parent F#570, distinct from the MEMENTO cluster which contained F#699/F#737/F#738/F#739). The **target-only-KC-panel-under-preempt-KILL micro-pattern canonicalizes at this observation** via cross-cluster triple-fire: 1st obs F#738 (behavioral target-only, MEMENTO), 2nd obs F#739 (engineering target-only, MEMENTO), 3rd obs this (engineering target-only, Pierre-serving). Cross-cluster independence strengthens the pattern beyond within-cluster repetition — promotes watchlist → canonical per mem-pattern-triple-fire.

## Predicted vs measured

| KC    | Prediction                                                                 | Measurement                                            | Outcome              |
| ----- | -------------------------------------------------------------------------- | ------------------------------------------------------ | -------------------- |
| K1911 | N=3 concurrent-stack throughput ratio < 50% (FAIL threshold for deployability) | **unmeasured** (preempt-blocked; ratio `NaN / NaN`)   | **FAIL (untested)**  |
| K1912 | Peak memory > 40GB at N=5 stacks (FAIL threshold for 48GB headroom)        | **unmeasured** (preempt-blocked; memory `NaN`)         | **FAIL (untested)**  |

All predictions match the structural expectation — the experiment correctly anticipates that current parent-state (F#570 KILLED) makes every KC unmeasurable. This is an honest preempt-KILL, not a "not yet tested" deferral.

## F#570 preconditions mapped onto this child

Inherited from F#570 (verified 2026-04-18 by source inspection of `mlx_lm/server.py:1155,1236` and filesystem state; re-inherited here):

| Precondition | F#570 state | This child K1911 | This child K1912 |
|--------------|-------------|------------------|------------------|
| **T1B** loader plugin API | ❌ no `mlx_lm.loaders/plugins/providers` entry-point group | blocks multi-adapter dispatch substrate | blocks multi-adapter dispatch substrate |
| **T1C** `pierre-g4e4b` checkpoint | ❌ 0 matches in `~/.cache/huggingface/hub`, no `micro/models/pierre-g4e4b/` | blocks base-model load → no N=1 baseline either | blocks base-model load → no N=5 residency |
| **T2** multi-adapter body schema | ❌ `mlx_lm/server.py:1236` validates `body["adapter"]` as single `str` | blocks per-request multi-adapter dispatch | blocks N≥2 resident-stack serving |
| **T3** adapter safetensors | ❌ `exp_p1_t2_single_domain_training/adapters/{math,code,medical}/adapters.safetensors` absent | blocks N=1 baseline | blocks N≥3 resident-set instantiation |
| **DEP** `exp_prod_pip_package_pierre` | ❌ KILLED | blocks headline `pip install pierre` CLI | blocks headline CLI |

## Theorems satisfied (per MATH.md)

- **T1 (preempt-transitivity):** Verified by parent F#570 KILL state + source inspection. A child KC that is a strictly stronger claim than its parent's KC (concurrent-multi-adapter serving strictly stronger than F#570's single-adapter loader) is preempt-blocked while the parent remains KILLED.
- **T2 (F#666 vacuous compliance):** Verified: K1911 and K1912 are both engineering targets (throughput-ratio and peak-memory-bytes ARE the targets); no pairable proxy exists that doesn't require the same parent-harness precondition. F#666 is satisfied trivially (the rule constrains proxy→target pairing; with zero proxies, the constraint is vacuous).
- **T3 (cross-cluster triple-fire for target-only panels):** Verified by recording 3 independent observations: F#738 (behavioral, MEMENTO), F#739 (engineering, MEMENTO), and this (engineering, Pierre-serving). Cross-cluster independence strengthens the pattern beyond within-cluster repetition; micro-pattern canonicalizes at 3rd obs.

## Why this is not "skip and rerun later"

The blockers are not configuration tweaks; they are:

- **Upstream feature gaps.** mlx-lm plugin-loader API is a multi-file change in someone else's project (F#570 T1B); body-schema extension from `str` to multi-adapter list is an upstream-repo PR (T2).
- **Produced-artifact gaps.** `pierre-g4e4b` composite and trained adapter safetensors are inputs this experiment consumes, not produces (T1C, T3).
- **Parent-extension scope.** Even at F#570 SUPPORTED, concurrent N≥5 residency + RSS/Metal-unified-pool instrumentation + ≥3-client concurrency driver are extensions beyond F#570's single-adapter loader scope.

Re-claiming requires all five F#570 preconditions + parent-extension. No KC-augmentation needed (both already engineering targets per F#666).

## Sub-axis classification

This is a **2-point serving-config spot-measurement** at N∈{3,5} with implicit N=1 baseline for K1911's ratio. Distinct from:

- **F#699 single-config** (one N, one metric).
- **F#737 canonical scalar-sweep** (one metric across many N).
- **F#738 categorical cross-corpus** (distinct corpora, behavioral metric).

Conservatively classified as its own variant ("serving-config spot-measurement"), **does NOT automatically advance** the multi-parent-run sub-axis counter (remains at 2 observations: F#737 + F#738). Canonical promotion of multi-parent-run sub-axis still pending a genuine 3rd same-metric-across-configs observation.

## Target-only-KC-panel-under-preempt-KILL: canonical promotion

Prior watchlist state (pre-this-experiment) tracked 2 MEMENTO-cluster observations:

1. **F#738** — target-only behavioral (accuracy ratio, MEMENTO).
2. **F#739** — target-only engineering (latency ×2, MEMENTO).

This experiment is the **3rd observation, cross-cluster** (Pierre-serving cluster, parent F#570). Cross-cluster independence is the strongest form of triple-fire: the micro-pattern is not confined to a single parent's idiosyncrasies.

**Canonical form:** an F#669 child whose KC panel is target-only (engineering OR behavioral) with no pairable proxy — satisfies F#666 by vacuous quantification rather than compound pairing. This is a legitimate F#666-compliance path; the rule's intent is that proxies not be asserted without target pairing, and target-only panels assert no proxy.

## Antipattern scan — see MATH.md §6/§7

All six rejected shortcuts documented in MATH.md §6: single-stack repeated serving, sequential N=1 timings summed as concurrent, untrained adapter-config shells, base-gemma substituted for pierre-g4e4b, vLLM/llama.cpp cross-framework substitution, analytical memory back-derivation. Each would constitute antipattern-t (silent objective swap) and/or antipattern-m (proxy-model-substitution).

## Assumptions logged

- F#570 preconditions (T1B/T1C/T2/T3/DEP) inherited from the 2026-04-18 verification; not re-probed this iteration since no MLX code path executes and the preempt-KILL verdict does not depend on the preconditions having changed state (if anything changed, parent status would have been updated from `killed` to `supported`, which it has not).
- `platform` field on the experiment record is `~` (null); preempt-KILL verdict does not depend on platform since no code executes. Hygiene note only.
- `success_criteria` field is empty (F#702 hygiene issue); noted but not patched this iteration since preempt-KILL supersedes hygiene correction per precedent (patchable later when experiment is eventually re-claimed).
