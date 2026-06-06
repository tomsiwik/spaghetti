# PAPER: Pierre adapter cache — idle-time pre-fill for first-token latency hiding

**Status:** KILLED — preempt-structural (F#669 13th reuse)
**Verdict line:** KILLED (no kill criterion measurable; parent `exp_prod_mlxlm_integration` is KILLED per F#570 with 5 preconditions unresolved; parent-extension requirements beyond F#570 scope; **not** PROVISIONAL — proven unmeasurable on current state)

## TL;DR

Two target-only engineering kill criteria (K1913 pre-fill TTFT reduction < 20%; K1914 cache memory overhead > 2GB) both FAIL as **unmeasured** due to parent-target-unverified preempt block. mlx-lm 0.31.2 has no plugin/loader API (F#570 T1B), no `pierre-g4e4b` checkpoint exists (T1C), server body schema validates `adapter` as single `str` not multi-adapter list (T2 — renders pre-fill a no-op since single adapter is always resident), trained adapter safetensors missing (T3), and grandparent `exp_prod_pip_package_pierre` is KILLED (DEP). Plus parent-extension requirements (idle-time hook, cache scheduler, TTFT cold-vs-warm instrumentation, cache-region memory probe) are beyond F#570's single-adapter loader scope — same structural pattern as sibling F#740.

This is the **13th F#669 reuse overall** and the **2nd Pierre-serving-cluster F#669 child preempt-KILL** (1st within-cluster reuse after F#740 `exp_pierre_multi_adapter_serving_throughput`). The **target-only-KC-panel-under-preempt-KILL micro-pattern is post-canonical**: canonicalized at F#740 via cross-cluster triple-fire (F#738 behavioral/MEMENTO + F#739 engineering/MEMENTO + F#740 engineering/Pierre-serving). This observation is the 4th and the 1st within-cluster reuse inside the Pierre-serving cluster — tally-only, confirms canonical form's within-cluster portability across distinct sub-axis variants (F#740 N-spot-measurement + this single-config idle-time pre-fill).

**Sub-axis classification:** single-config idle-time pre-fill (2 engineering metrics on one configuration) — **2nd observation of single-config-target-only-engineering sub-axis variant** (1st obs was F#739 in MEMENTO cluster). Cross-cluster reuse of this sub-axis variant. Does NOT advance the canonical multi-parent-run sub-axis counter (remains at 2 obs: F#737 + F#738). 1 more distinct obs would canonicalize the single-config-target-only-engineering variant.

## Predicted vs measured

| KC    | Prediction                                                                       | Measurement                                                 | Outcome              |
| ----- | -------------------------------------------------------------------------------- | ----------------------------------------------------------- | -------------------- |
| K1913 | Pre-fill TTFT reduction < 20% (FAIL threshold for user-perceived responsiveness) | **unmeasured** (preempt-blocked; both cold and warm TTFT `NaN`) | **FAIL (untested)**  |
| K1914 | Cache memory overhead > 2GB (FAIL threshold for KV-cache coexistence headroom)   | **unmeasured** (preempt-blocked; overhead `NaN`)           | **FAIL (untested)**  |

All predictions match the structural expectation — the experiment correctly anticipates that current parent-state (F#570 KILLED) plus parent-extension gap makes every KC unmeasurable. This is an honest preempt-KILL, not a "not yet tested" deferral.

## F#570 preconditions mapped onto this child

Inherited from F#570 (verified 2026-04-18 by source inspection of `mlx_lm/server.py:1155,1236` and filesystem state; re-inherited here since parent state has not changed since F#740 confirmed it):

| Precondition | F#570 state | This child K1913 | This child K1914 |
|--------------|-------------|------------------|------------------|
| **T1B** loader plugin API | ❌ no `mlx_lm.loaders/plugins/providers` entry-point group | blocks multi-adapter dispatch substrate pre-fill cache hooks into | blocks multi-adapter dispatch substrate |
| **T1C** `pierre-g4e4b` checkpoint | ❌ 0 matches in `~/.cache/huggingface/hub`, no `micro/models/pierre-g4e4b/` | blocks base-model load → no TTFT baseline | blocks base-model load → no cache-delta reference |
| **T2** multi-adapter body schema | ❌ `mlx_lm/server.py:1236` validates `body["adapter"]` as single `str` | single-adapter schema makes pre-fill a no-op (single adapter always resident) | blocks multi-adapter identity set that pre-fill scheduler requires |
| **T3** adapter safetensors | ❌ `exp_p1_t2_single_domain_training/adapters/{math,code,medical}/adapters.safetensors` absent | blocks cold-path baseline and warm-path cache-hit | blocks cache-pre-fill content |
| **DEP** `exp_prod_pip_package_pierre` | ❌ KILLED | blocks headline `pierre serve --cache-prefill on` CLI | blocks headline CLI |

## Parent-extension requirements (beyond F#570 scope)

Even if all 5 F#570 preconditions resolve (parent SUPPORTED), K1913/K1914 remain strictly stronger and require parent-extension at the Pierre-serving-infrastructure layer:

| Extension | Required for | Scope |
|-----------|--------------|-------|
| Idle-time hook in serve-loop | K1913 (defines when pre-fill may run) | `on_idle` / `before_next_request` / asyncio-scheduler callback |
| Adapter-cache layer with pre-fill policy | K1913, K1914 | next-likely-adapter hypothesis + eviction strategy (LRU / explicit-pin / heuristic) |
| TTFT instrumentation (cold vs warm) | K1913 | distinguishes cache-miss (cold) from cache-hit (warm) paths |
| Memory-overhead probe (cache region isolation) | K1914 | separates cache region from base model and KV cache for clean attribution |

## Theorems satisfied (per MATH.md)

- **T1 (preempt-transitivity):** Verified by parent F#570 KILL state + source inspection. A child KC that is a strictly stronger claim than its parent's KC (adapter-cache-prefill-during-idle strictly stronger than F#570's single-adapter loader) is preempt-blocked while the parent remains KILLED. Additionally, parent-extension (idle-time hook + cache scheduler + TTFT and memory-delta instrumentation) is required beyond parent's scope.
- **T2 (F#666 vacuous compliance):** Verified: K1913 and K1914 are both engineering targets (TTFT-reduction-% and memory-overhead-GB ARE the targets); no pairable proxy exists that doesn't require the same parent-harness precondition. F#666 is satisfied trivially (the rule constrains proxy→target pairing; with zero proxies, the constraint is vacuous).
- **T3 (target-only-KC-panel post-canonical reuse):** Recorded as 4th observation of the canonical pattern (post-F#740). 1st within-cluster reuse inside the Pierre-serving cluster — confirms canonical form's within-cluster portability across distinct sub-axis variants (F#740 N-spot-measurement + this single-config idle-time pre-fill).

## Why this is not "skip and rerun later"

The blockers are not configuration tweaks; they are:

- **Upstream feature gaps.** mlx-lm plugin-loader API is a multi-file change in someone else's project (F#570 T1B); body-schema extension from `str` to multi-adapter list is an upstream-repo PR (T2).
- **Produced-artifact gaps.** `pierre-g4e4b` composite and trained adapter safetensors are inputs this experiment consumes, not produces (T1C, T3).
- **Parent-extension scope.** Even at F#570 SUPPORTED, idle-time hook + cache scheduler + TTFT cold-vs-warm instrumentation + cache-region memory probe are extensions beyond F#570's single-adapter loader scope.

Re-claiming requires all five F#570 preconditions + parent-extension. No KC-augmentation needed (both already engineering targets per F#666).

## Sub-axis classification

This is a **single-config idle-time pre-fill** measurement (2 engineering metrics — K1913 TTFT reduction, K1914 cache memory overhead — on one configuration). Distinct from:

- **F#699 single-config** (one N, one metric; proxy+target).
- **F#737 canonical scalar-sweep** (one metric across many N).
- **F#738 categorical cross-corpus** (distinct corpora, behavioral metric).
- **F#739 single-config engineering** (one config, 2 engineering metrics — **SAME sub-axis variant as this**).
- **F#740 serving-config spot-measurement at N∈{3,5}** (distinct metrics at different N; Pierre-serving cluster sibling).

Classification: **single-config target-only engineering (2 distinct metrics on one config)** — the same sub-axis variant as F#739. This is the **2nd observation of single-config-target-only-engineering sub-axis variant** (1st was F#739 in MEMENTO cluster). Cross-cluster reuse of this sub-axis variant. Does NOT advance the canonical multi-parent-run sub-axis counter (remains at 2 obs: F#737 + F#738). 1 more distinct obs would canonicalize the single-config-target-only-engineering variant per the triple-fire rule.

## Target-only-KC-panel-under-preempt-KILL: post-canonical reuse

Pattern canonicalized at F#740 via cross-cluster triple-fire (F#738 + F#739 + F#740). This experiment is the **4th observation** and the **1st within-cluster reuse inside the Pierre-serving cluster**. Tally only; no new canonicalization event.

**Strengthening contribution:** within-cluster reuse across distinct sub-axis variants (F#740 N-spot-measurement + this single-config idle-time pre-fill) confirms the canonical form's portability beyond cross-cluster independence — the pattern is cluster-portable AND sub-axis-portable within the same cluster.

## Pierre-serving-cluster consolidation (watchlist)

With 2 Pierre-serving F#669 children (F#740 + this) and remaining open P≤2 children still tagged `serving`/`p1` likely to preempt-KILL identically, **consolidation into a single cluster-level finding is a reasonable reviewer option** for subsequent Pierre-serving children. This experiment still files its own F#669 13th-reuse finding; future children may be resolved as a batch under one consolidated finding instead.

## Antipattern scan — see MATH.md §6/§7

All six rejected shortcuts documented in MATH.md §6: no-cache-layer single-stack TTFT substitution, startup-preload single-adapter substitution, os.fork-COW as "cache", base-gemma-without-Pierre-wrapper, torch OS-page-cache prefetch, analytical memory back-derivation. Each would constitute antipattern-t (silent objective swap) and/or antipattern-m (proxy-model-substitution).

## Assumptions logged

- F#570 preconditions (T1B/T1C/T2/T3/DEP) inherited from the 2026-04-18 verification + F#740 2026-04-24 confirmation; not re-probed this iteration since no MLX code path executes and the preempt-KILL verdict does not depend on the preconditions having changed state (if anything changed, parent status would have been updated from `killed` to `supported`, which it has not).
- `platform` field on the experiment record is `~` (null); preempt-KILL verdict does not depend on platform since no code executes. Hygiene note only.
- `success_criteria` field is empty (F#702 hygiene issue); noted but not patched this iteration since preempt-KILL supersedes hygiene correction per precedent (patchable later when experiment is eventually re-claimed).
- Parent-extension requirements (idle-time hook, cache scheduler, TTFT cold-vs-warm instrumentation, cache-region memory probe) are enumerated from the mechanism-under-test, not from a pre-existing design document for Pierre's pre-fill layer (which does not exist).
