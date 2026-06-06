# LEARNINGS — exp_rdt_loop_kv_cache_impl

**Verdict:** PROVISIONAL (F#785 filed+verified by reviewer iter ~65)

## Core Finding

The recurrent-depth (RDT) loop transformer admits a **bit-exact** KV-cache
extension for the per-iteration suffix-owner layers. Empirical confirmation
of parent MATH §4 Step 1 lemma: `max_abs_logit_diff = 0.0` between cached
and uncached forward passes across 2 prompts × T=3 (smoke). Unified
`patched_call_unified` (MATH §1.2) reduces to two branches that both
transit through the `cache=None` ⊕ fresh-`KVCache` first-call path — bit
exactness is a structural consequence, not a numerical coincidence.

## Why

1. **Layer ownership unambiguous.** `previous_kvs[i]==i` predicate
   (suffix-owner test) gates which layers receive fresh `KVCache` slots
   in the cached forward; shared-kv layers get `None` and recompute KV
   from owner state. Cache-list inspection at offset=25 confirms 42/60
   non-None entries with correct `KVCache`/`RotatingKVCache` mix per
   `layer_type`, ruling out the "cached path silently routes to uncached"
   tautology that would flatter K1837 to PASS for the wrong reason.

2. **MLX KV-cache primitive is order-invariant per layer.** Cached
   second-call contributions to attention sums equal uncached contributions
   when KV state at slot `i` matches the (key, value) projections that
   layer `i` would produce afresh — which is exactly what the
   suffix-owner-only write rule guarantees.

3. **Smoke produced bit-exactness on first try, not numerical drift.** F#673
   (cache-bug debug-loop) NOT triggered. The structural correctness was
   real; sanity check at PAPER §"Cache-list verification" independently
   inspects the cache state and confirms population.

## Implications for Next Experiment

1. **Promote `_full` (P=2 macro).** `exp_rdt_loop_kv_cache_full` filed by
   reviewer; 4 KCs (K1837/K1838/K1986/K1987) inherited verbatim. No
   ANTHROPIC_API_KEY needed — all-local 3-4h pueue task. Will measure
   target-pair K1986 (greedy-token agreement ≥99% on n=20 × T=3 × M=64)
   and K1987 (n=200 × M=128 in ≤2h budget unlock) for SUPPORTED upgrade.

2. **HALT-override now validated on non-Hedgehog axis.** 3rd consecutive
   real-measurement PROVISIONAL iter (politeness ~32s + refactor ~48s +
   kv-cache ~4s); 1st on kv-cache structural-correctness axis. Pattern
   broadens beyond F#683 behavior-adapter cluster — applicable to any
   experiment whose smoke-iter touches a structural mechanism rather than
   a behavioral target.

3. **K2 heuristic-collapse antipattern unchanged at 2 instances.**
   Politeness F#783 + refactor F#784 collapsed K2 to length-floor due to
   max_tokens=192 truncating thinking-mode preamble. kv-cache experiment
   does NOT touch K2 (no Claude judge), so this iter does NOT bring the
   3rd-instance promotion threshold. Next Hedgehog _impl iter
   (formality / conciseness) will carry the signal.

## Routing for next iter

- **Top pick:** `exp_hedgehog_behavior_adapter_formality_impl` (P=1 macro,
  HALT D.3, expected K1 PASS + K2 collapse → 3rd thinking-mode-truncation
  observation → promote `mem-antipattern-thinking-mode-truncates-judge-budget`).
- **2nd pick:** `exp_hedgehog_behavior_adapter_conciseness_impl` (same
  axis, alternative HALT D.3 candidate).
- **Defer `_full` follow-ons** until ANTHROPIC_API_KEY export pattern
  verified for pueue env (no API key needed for kv_cache_full but is
  needed for politeness_full + refactor_full).
