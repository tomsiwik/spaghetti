# PAPER — exp_rdt_loop_kv_cache

**Verdict: PROVISIONAL (macro-scope design-only, `_impl` at P3).**

## Summary

Design of a KV-cache-aware recurrent-depth forward for the parent
`exp_rdt_loop_lora_gemma4_bench` loop-LoRA stack on
`mlx-community/gemma-4-e4b-it-4bit`. Parent's Theorem 2(b) identified
this as the single structural unlock to move K1740-BENCH from
`under_powered` (n=30 << 200) to measurable; parent-estimated
uncached eval at n=200 T=3 costs ~183h on M5 Pro — infeasible without
cache.

The experiment produces a formal mathematical design (MATH.md §1-§4)
of the cache layout: T caches per looped layer, total cache list
length `33 + 9T`. Bit-exact equivalence theorem proved under exact
arithmetic; fp16 error bound ≤ 1e-3 derived. Empirical verification
is scope-deferred to `exp_rdt_loop_kv_cache_impl` at P3 per
reviewer.md §5 PROVISIONAL-as-design macro-scope clause and handoff
instruction #4.

## Prediction vs measurement

| KC | Prediction | Measurement | Status |
|---|---|---|---|
| K1764 (bit-exact) | max_abs_logit_diff < 1e-3 across all 80 (T, prompt) pairs | **not measured** — scaffold only | `not_measured` |
| K1765 (5× speedup) | cached T=3 ≥ 5× faster than uncached on n=20, M=64 | **not measured** — scaffold only | `not_measured` |

Scaffold `run_experiment.py` elapsed 4.31s; no forward passes
executed. `results.json` lists both KCs as `not_measured` with
explicit unblock pointer to `_impl` at P3.

## What this experiment produces

1. **Cache layout primitive (new).** `33 + 9T` cache list with the
   index formula:
   - Non-loop prefix: `cache[i]` for `i ∈ [0, LOOP_START)`.
   - Loop region: `cache[LOOP_START + t · N_LOOP + (j - LOOP_START)]`
     for `t ∈ [0, T), j ∈ [LOOP_START, LOOP_END)`.
   - Non-loop suffix: `cache[LOOP_START + T · N_LOOP + (i - LOOP_END)]`
     for `i ∈ [LOOP_END, L)`.
   Per-entry: `KVCache()` for full-attention, `RotatingKVCache(
   max_size=sliding_window, keep=0)` for sliding-attention.

2. **Patched `__call__` pseudocode (MATH.md §1.3).** Threads
   cache-index into the nested loop via explicit computation; leaves
   `previous_kvs` sharing for non-loop suffix layers unchanged.

3. **Bit-exact equivalence theorem (MATH.md §4).** In exact
   arithmetic, `h_unc == h_cached` element-wise. In fp16, the bound
   is `< 1e-3` derived from ~96 accumulation ops at 2^-10 ULP each.

4. **Caveat on RotatingKVCache (MATH.md §4).** Restriction to
   prompts ≤ 400 tokens keeps cache length < 512 (sliding_window),
   avoiding truncation-history divergence.

5. **Flag on parent's latent bug (MATH.md §1.1).** The parent code
   passes `cache=cache[j]` inside the loop — if `cache[j]` were
   non-None (parent ran with cache=None throughout), the cache would
   accumulate T different (K,V) concatenated per original token. The
   parent's `SKIP_KVCACHE=1` default avoids this; the bug is dormant.
   `_impl` must not reuse parent's cache-threading verbatim.

## Why PROVISIONAL-as-design

Per MATH.md §6:
- Mathematical construction complete (Theorem + fp16 bound + cache
  formulas + pseudocode).
- Empirical budget ~3-4h plausibly exceeds researcher-hat 2h cap:
  80 forward pairs at Gemma 4 E4B uncached speed (~1 min each on
  short prompts) = ~80 min; plus cache-bug debug risk (F#673
  lineage, KV-cache bugs produce silent wrong logits).
- Parent's reference runtime was 6644s (1.8h) on a simpler workload.
- Per reviewer.md §5 "PROVISIONAL (macro-scope design-only
  sub-case)" clause: design + scaffold + `_impl` at P3 is the
  canonical routing when empirical scope exceeds single-iter budget.

## Assumptions

- **L=42** (Gemma 4 E4B num_hidden_layers). Formulas are symbolic in
  L; correct for any L.
- **first_kv_shared=22** (num_kv_shared_layers=20). Formula in §1.2
  assumes all looped layers (indices 12..20) own their cache —
  requires `first_kv_shared ≥ LOOP_END=21`. For Gemma 4 E4B this
  holds (22 ≥ 21).
- **Prompt length ≤ 400 tokens**. GSM8K-valid empirical mean ~120,
  95%ile ~300. MATH.md §4 caveat details.
- **mlx==0.31.1, mlx-lm==0.31.2**. Pinned to parent for API-surface
  compatibility; `Attention.__call__` signature with `shared_kv=`,
  `offset=`, `cache.update_and_fetch(K, V)` contract, and
  `Gemma4TextModel.make_cache` location are all read from these
  exact versions.

## Antipattern audit (summary; full table in MATH.md §7)

All 12 antipattern categories checked. Zero triggers. Key defences:
- Antipattern (t) — scope-swap — not triggered. §0 F1-F6 lock is
  preserved verbatim; unverified KCs are reported `not_measured`.
- Antipattern (m) — proxy-model — not triggered. Scaffold loads
  (conditionally imports) the exact base model
  `mlx-community/gemma-4-e4b-it-4bit` per §0 F1.
- Antipattern `smoke-as-full` — not triggered. `is_smoke=false`;
  verdict PROVISIONAL, not supported.

## Unblock

- **Primary:** `exp_rdt_loop_kv_cache_impl` at P3 inherits K1764 +
  K1765 verbatim. Implements MATH.md §1.3 pseudocode and runs the
  full 80-pair K1764 verification + 20-prompt K1765 speedup test.
  Expected empirical budget 3-4h; runs as P3 with explicit budget.
- **Secondary:** Once K1764 + K1765 both PASS, `exp_rdt_loop_kv_cache`
  lifts from PROVISIONAL to SUPPORTED via evidence-add from `_impl`.
- **Downstream:** Parent `exp_rdt_loop_lora_gemma4_bench` (F#674,
  PROVISIONAL) can move PROVISIONAL→SUPPORTED once K1740/K1741/K1742
  are measured at pre-registered n — feasible once K1765 unlocks
  the 5× speedup for eval at n=200.

## References

- Parent: `exp_rdt_loop_lora_gemma4_bench` MATH.md §Theorem 2(b);
  results.json K-KVCACHE entry; Finding #674 caveat.
- Parent-parent: `exp_rdt_loop_lora_gemma4_full` (Finding #667, #673).
- Bae 2024, arxiv:2410.20672 (Looped Transformers; no KV-cache
  treatment for looped generation).
- Saunshi 2025, arxiv:2502.17416 (Recurrent-depth reasoning; OOS for
  inference efficiency).
- mlx_lm 0.31.2: `models/cache.py` KVCache + RotatingKVCache source;
  `models/gemma4_text.py` Model.make_cache + Gemma4TextModel.__call__
  source.
