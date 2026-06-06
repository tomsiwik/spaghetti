# MATH — exp_rdt_loop_kv_cache_impl

Empirical companion to `exp_rdt_loop_kv_cache` (PROVISIONAL). Implements
the MATH.md §1.3 cache-list pseudocode and runs the K1764/K1765 verification
that was scope-deferred from the parent design experiment per its §6
PROVISIONAL-as-design clause.

## §0 Skills invoked, scope lock, pinned versions

Platform skills `/mlx-dev` + `/fast-mlx` invoked before writing this
document and the companion `run_experiment.py`, per PLAN.md Part 2 +
guardrail 1012 (the largest single source of broken code in this repo's
history per audit). Items internalized and applied:

- `mx.eval(...)` only at loop boundaries; MLX is lazy. Avoid per-op eval.
- Slices create copies (not views — opposite of NumPy). Use `mx.concatenate`
  when extending cache state.
- `mx.array([...], dtype=mx.int32)` for tokenizer outputs (default float
  dtype promotion is wrong for token ids).
- `cache.update_and_fetch(K, V)` on a fresh `KVCache` returns `(K, V)`
  exactly (parent MATH §4 Step 1; mlx_lm KVCache source verified).
- Avoid `.item()` inside hot loops — implicit eval. Capture as `mx.array`,
  evaluate once at end.
- `mlx==0.31.1`, `mlx-lm==0.31.2` pinned (parent-inherited; Gemma4TextModel
  monkey-patch surface read at this version).

**Scope-preservation lock (reviewer antipattern (t) defence) — F1–F6
inherited verbatim from parent MATH §0:**

- F1: Base = `mlx-community/gemma-4-e4b-it-4bit`.
- F2: Loop region = layers 12..20 inclusive (9 layers; LOOP_START=12,
  LOOP_END=21, N_LOOP=9).
- F3: T-sweep = {1, 2, 3, 6} for K1837 + K1838.
- F4: n=20 prompts (GSM8K-valid subset) for K1837 + K1838; n=200 for K1987.
- F5: Bit-exact tolerance = max_abs_logit_diff < 1e-3 in fp16 (K1837).
- F6: Speed threshold = cached ≥ 5× faster than uncached, wall-clock
  matched n + matched M new tokens (K1838).

**SMOKE-MODE relaxation (this iter only).** Smoke runs reduce n to 2
prompts and restrict T to {3} only — explicitly to land K1837 with real
measurement under researcher-hat budget. K1838/K1986/K1987 are deferred
to `exp_rdt_loop_kv_cache_full` (P=2 macro). Smoke verdict ceiling is
PROVISIONAL per PLAN §1010 #4. **Not a scope swap**: F1–F6 remain the
canonical scope; the smoke is explicitly marked `is_smoke=true` and
labelled "preliminary" in PAPER.md. If F1–F6 must change for tractability,
this is filed as a follow-up scope-deferral with reasoning, never silent.

## §1 Architecture — cache layout under recurrent-depth forward

Inherits parent MATH §1.1–§1.4 verbatim. Three artefacts implemented in
`run_experiment.py`:

### 1.1 Cache-index formula (parent §1.2, recap)

For L=42, LOOP_START=12, LOOP_END=21, N_LOOP=9, T=T_ref[0]:

- Non-loop prefix: `c_idx = i` for `i ∈ [0, LOOP_START)`. 12 entries.
- Loop region: `c_idx = LOOP_START + t · N_LOOP + (j - LOOP_START)` for
  `t ∈ [0, T)`, `j ∈ [LOOP_START, LOOP_END)`. T·9 entries.
- Non-loop suffix: `c_idx = LOOP_START + T·N_LOOP + (i - LOOP_END)` for
  `i ∈ [LOOP_END, L)`. 21 entries.

Total length = `12 + T·9 + 21 = 33 + 9T`. T=3 ⇒ 60. T=6 ⇒ 87.

### 1.2 Unified patched `__call__` (this experiment's IMPL choice)

Where parent MATH §1.3 prescribed two distinct call signatures (uncached
parent's; cached new), we collapse to a **single unified
`patched_call_unified`** that detects cache-list length:

```python
def patched_call_unified(self, inputs=None, cache=None, ...):
    h = self.embed_tokens(inputs) * self.embed_scale
    # ... per_layer_input prep (parent-inherited verbatim) ...
    expected_len = LOOP_START + T_ref[0] * N_LOOP + (len(self.layers) - LOOP_END)
    if cache is None:
        cache = [None] * expected_len
    elif len(cache) < expected_len:
        cache = cache + [None] * (expected_len - len(cache))
    # masks indexed by underlying layer (length L); use cache slice that
    # matches per-layer offsets (loop layers use last-iter cache for
    # mask computation).
    mask_cache = _build_mask_cache_alias(cache, T_ref[0])
    masks = self._make_masks(h, mask_cache)
    # ... rest unchanged from parent §1.3 ...
```

**Why unified, not two functions.** The c_idx formula degrades
gracefully: when `cache=None → cache=[None]*expected_len`, every
`cache[c_idx]=None`, and mlx_lm's attention path with
`cache=None` is bit-identical to the uncached parent path (per
parent MATH §4 Step 1 lemma). So a single function with a single
formula handles both branches, eliminating the risk of two
divergent code paths. This is an IMPL-grade tightening; the
mathematical guarantee in parent MATH §4 transfers verbatim.

### 1.3 `_build_mask_cache_alias`

Maps the length `33 + 9T` cache list back to a length-L mask
companion list:

```python
def _build_mask_cache_alias(cache, T_now):
    L = num_hidden_layers
    out = []
    for i in range(L):
        if i < LOOP_START:
            out.append(cache[i])
        elif i < LOOP_END:
            # Use the LAST loop-iteration's cache for layer i's mask
            # computation. Mask depends on cache offset (sliding-window
            # truncation); last-iter is the maximum-offset case.
            out.append(cache[LOOP_START + (T_now - 1) * N_LOOP + (i - LOOP_START)])
        else:
            out.append(cache[LOOP_START + T_now * N_LOOP + (i - LOOP_END)])
    return out
```

For fresh caches (offset=0), mask shape is determined by `h` length only,
so the alias-mapping is trivially correct. For non-fresh caches (mid-gen),
the alias picks the largest offset per loop layer (last iteration) — this
is correct because (i) all T loop iterations advance their caches by the
same number of tokens at each forward (one per generation step), so all
loop-iter caches at layer j have the same offset; (ii) attention masks
in the loop region depend on the *common* offset, not per-iter offsets.

## §2 Prior art / dependencies

- Parent design: `exp_rdt_loop_kv_cache` (PROVISIONAL, F#690 + cache-list
  pseudocode locked). This impl inherits MATH §0–§4 verbatim.
- Parent's parent (the looped LoRA bench): `exp_rdt_loop_lora_gemma4_bench`
  (F#674 PROVISIONAL — structural+dynamic PASS at n<200; this impl unblocks
  K1740 measurement at pre-reg n).
- mlx_lm KV-cache contract (`KVCache.update_and_fetch`): parent §2.2.

## §3 Kill criteria (pre-registered, locked)

Inherits parent K1764 → K1837 (#1837 in DB) verbatim, parent K1765 →
K1838 (#1838 in DB) verbatim. Adds F#770-schema-repair target pairs:

- **K1837** (proxy, mechanism correctness): max_abs_logit_diff < 1e-3 on
  n=20 GSM8K prompts × T ∈ {1,2,3,6} = 80 pairs.
- **K1838** (proxy, speedup): cached T=3 ≥ 5× faster than uncached on
  n=20 prompts × M=64 new tokens.
- **K1986** (target, pairs K1837 per F#666): greedy-token agreement rate
  ≥ 99% between cached and uncached gen on n=20 GSM8K-Hard prompts
  at T=3, M=64. Behavioral parity check; logit-diff is correctness, this
  is downstream-task parity.
- **K1987** (target, pairs K1838 per F#666): cached T=3 generation
  completes n=200 GSM8K-Hard at M=128 within 2h researcher-hat wall-clock
  budget — the explicit unlock claimed by parent MATH §4 (uncached takes
  ~183h per parent's Theorem 2 derivation).

### 3.1 Smoke-mode KC subset

Smoke completes K1837 only at reduced n=2 × T=3 (6 logit-pair
measurements). All other KCs are explicitly `not_measured` with
follow-up reason logged. Smoke verdict ceiling: PROVISIONAL.

## §4 Theorem inheritance

Bit-exact equivalence theorem (parent MATH §4) transfers verbatim because
the unified §1.2 patched call uses the same c_idx formula in both
branches; the only branch difference is whether `cache[c_idx]` is `None`
(uncached) vs a fresh `KVCache()` (cached). Per parent §4 Step 1,
`KVCache().update_and_fetch(K, V)` on first call returns exactly `(K, V)`
— bit-identical to mlx_lm's `cache=None` branch which uses `(K, V)`
directly. So `h_unc == h_cached` bit-exactly per the inductive proof.

## §5 Prediction vs measurement

| KC | Prediction (smoke) | Prediction (full) | Measurement path |
|---|---|---|---|
| K1837 | max_abs_logit_diff < 1e-3 on 6 pairs (n=2, T=3) | < 1e-3 on 80 pairs | `verify_bit_exact()` |
| K1838 | not_measured (smoke skip) | cached T=3 ≥ 5× faster | `verify_speedup()` |
| K1986 | not_measured (smoke skip) | greedy-token agreement ≥ 99% on n=20 T=3 | `verify_behavioral_parity()` |
| K1987 | not_measured (smoke skip) | cached n=200 M=128 ≤ 2h wall-clock | `verify_unlock_budget()` |

## §6 Antipattern self-audit

| Code | Antipattern | Status |
|---|---|---|
| composition-bug | `ΣA_i, ΣB_i` independently | n/a — single-adapter stack per forward |
| tautological-routing | `route(val[d][0])` | n/a — no routing |
| lora-scale-20 | `LORA_SCALE ≥ 12` | α=2 → scale=0.125 (parent-inherited, safe) |
| shutil-copy-adapter | `shutil.copy` | n/a |
| hardcoded-pass | `{"pass": True}` literal | run_experiment.py computes `result` from threshold comparison; never literal-pass |
| thinking-truncation | base eval avg_thinking_chars=0 | n/a — no generation eval in smoke |
| proxy-model | MATH says X, code loads Y | MATH = `mlx-community/gemma-4-e4b-it-4bit`, code loads same |
| smoke-as-full | `is_smoke=true` in full-run | smoke writes `is_smoke=true`, verdict=PROVISIONAL; full mode (SMOKE_TEST=0) writes `is_smoke=false` |
| kc-tautological | K passes by algebraic identity | K1837 tests max_abs_logit_diff against fp16 ULP-bounded threshold; non-trivial |
| kc-swap | KC modified after first result | K1837/K1838/K1986/K1987 locked verbatim from parent |
| copy-paste-scaffolding | sibling reused without re-reading | parent's `run_experiment.py` (bench) read line-by-line; only code patterns ported, not direct file reuse |
| scope-swap | silent change of base model / data / KC thresholds | §0 F1–F6 locked; smoke explicitly marked, never silent |

## §7 Libraries / pins

- `mlx==0.31.1`, `mlx-lm==0.31.2` (parent-pinned, 0.31.2 has the
  `previous_kvs` + `shared_kv=` API; do not upgrade in this iter).
- `datasets` deferred to `_full` (smoke uses 2 hand-coded short prompts).
- Seed 42 throughout. `mx.linalg.qr(stream=mx.cpu)` if QR needed (it is,
  for parent-inherited LoopLoRA init).

## §8 Assumptions logged (delta over parent)

- **Smoke prompt length**: hand-coded prompts ≤ 50 tokens. Below all
  sliding-window thresholds (parent §4 caveat). Trivially verified.
- **Mask-alias correctness for fresh cache**: trivially correct; verified
  by the bit-exact comparison itself (if the mask alias mismatched,
  K1837 would FAIL).
- **Pueue submission via SMOKE_TEST=1**: ~1-2 min budget. If pueue task
  exceeds 5 min, kill and re-design (F#673 cache-bug debug-loop signal).
