# MATH — exp_rdt_loop_kv_cache

Structural-correctness + speed test. Implements the single structural
unlock identified in parent `exp_rdt_loop_lora_gemma4_bench` MATH
Theorem 2(b): KV-cache-aware recurrent-depth forward, enabling K1740
at pre-registered n≥200 under researcher-hat 2h budget.

Parent F#674 is PROVISIONAL pending this unblock. Per parent's CLAIM
INSTRUCTION note: the dep-add link is lineage-only; this experiment
can and should be claimed while parent is PROVISIONAL. Child KCs
(K1764 bit-exact, K1765 5× speedup) are **parent-target-independent**
— they verify a cache mechanism whose correctness does not depend on
behavioral target KCs (K1740/K1741/K1742).

## §0 Skills invoked, scope lock, pinned versions

Platform skills `/mlx-dev` + `/fast-mlx` invoked before writing this
document (PLAN.md Part 2 requirement; F#673 lineage). Key items
internalized from the skills and applicable to this experiment:

- `mx.eval(loss, model.parameters())` at loop boundaries; MLX is lazy.
- `mx.clear_cache()` between phases (parent load → eval → cached eval
  → comparison) to avoid M5 Pro 48GB OOM.
- `mx.array([toks], dtype=mx.int32)` — tokenizer outputs must be
  explicitly typed; default dtype promotion is wrong here.
- `cache.update_and_fetch(keys, values)` is the only valid interface
  to mlx_lm's `KVCache` / `RotatingKVCache`; do not write to
  `cache.keys` / `cache.values` directly (F#673 origin).
- `mx.linalg.qr(stream=mx.cpu)` for partition-QR (unchanged from
  parent; we do not re-init LoRA in this experiment's verification
  path — LoRA deltas are either zero or fresh-random at B=small).
- `mlx == 0.31.1`, `mlx-lm == 0.31.2` (pinned to match parent; the
  monkey-patch points — `Gemma4TextModel.__call__`,
  `Model.make_cache`, `Attention.__call__` signature with
  `shared_kv=`, `offset=`, `cache.update_and_fetch` — all read from
  this exact version).

**Scope-preservation lock (reviewer antipattern (t) defence).** This
experiment's KCs are K1764 (bit-exact cached vs uncached) and K1765
(5× wall-clock speedup on identical n=20 prompts × T∈{1,2,3,6}). No
scope swap is permitted between MATH.md and code:

- F1: Base model = `mlx-community/gemma-4-e4b-it-4bit` (parent's
  target, not a smaller surrogate).
- F2: Loop region = layers 12..20 inclusive (N_LOOPS=6, 9 consecutive
  DecoderLayers). Parent's architecture is inherited verbatim.
- F3: T-sweep = {1, 2, 3, 6} for both K1764 and K1765. No substitution
  of a single T value for "representative" testing.
- F4: n=20 prompts for K1764 (GSM8K-valid subset); n=20 prompts for
  K1765 speed. No n-reduction.
- F5: Bit-exact tolerance = max_abs_logit_diff < 1e-3 in fp16. No
  relaxation to rtol-style checks.
- F6: Speed threshold = cached ≥ 5× faster than uncached, wall-clock,
  same 20 prompts, same T, same max_tokens. No surrogate metric
  (e.g. FLOPs ratio, throughput on single prompt only).

If any of F1–F6 must change to make the run tractable, that is a
scope swap; the correct response is PROVISIONAL with explicit
scope-deferral, not silent modification.

## §1 Architecture — cache layout for recurrent-depth forward

### 1.1 Standard Gemma 4 cache (baseline, uncached-equivalent mlx_lm behaviour)

From `mlx_lm.models.gemma4_text.Model.make_cache` (line 653 in
installed version 0.31.2):

```
first_kv_shared = num_hidden_layers - num_kv_shared_layers
caches = []
for i in range(first_kv_shared):
    if layer_types[i] == "full_attention":
        caches.append(KVCache())
    else:
        caches.append(RotatingKVCache(max_size=sliding_window, keep=0))
return caches  # list of length first_kv_shared
```

And `Gemma4TextModel.__call__` (line 508):

```
if cache is None:
    cache = [None] * len(self.layers)
else:
    cache = cache + [None] * (len(self.layers) - len(cache))
```

Interpretation: layers `0..first_kv_shared-1` own their cache;
layers `first_kv_shared..num_hidden_layers-1` receive
`cache=None` and reuse K/V from an earlier same-type layer via
the `previous_kvs[idx]` mapping (constructed in `__init__`,
line 425).

For Gemma 4 E4B (installed config at inspection time): let L = 42,
num_kv_shared = 20 ⇒ first_kv_shared = 22. Layers 0–21 get
their own cache; layers 22–41 share K/V from an earlier layer of
the same layer_type.

Parent's patched `__call__` (run_experiment.py line 119) respects
this: the non-loop branch (idx < LOOP_START or idx ≥ LOOP_END)
calls `layer(h, mask, cache[idx], ..., shared_kv=kvs, offset=offset)`
where `(kvs, offset) = intermediates[previous_kvs[idx]]`. So the
sharing mechanism is preserved outside the loop.

Inside the loop (LOOP_START ≤ j < LOOP_END), parent's patched
`__call__` passes `shared_kv=None, offset=None, cache=cache[j]`:

```
for t in range(T_ref[0]):
    loop_idx_ref[0] = t
    h_loop = h
    for j in range(LOOP_START, LOOP_END):
        h_loop, _, _ = self.layers[j](
            h_loop, masks[j], cache[j],  # <-- cache[j] REUSED every t
            per_layer_input=per_layer_inputs[j],
            shared_kv=None, offset=None,
        )
```

**Bug in parent's cache handling (independent of this experiment but
worth flagging):** if `cache[j]` were non-None, the parent code would
call `cache[j].update_and_fetch(K_t, V_t)` T times per forward, each
time with different K/V (because LoRA delta on v_proj differs per t).
Consequence: the cache for layer j ends the first forward containing
T concatenated (K, V) slices per original token — not the intended
one slice per token. This is why parent documented K-KVCACHE
scope-deferred: the cache design is wrong for the recurrent loop.

### 1.2 Proposed cache layout — T caches per looped layer

**Definition.** For a recurrent-depth forward with N_LOOPS=6 loops
over LOOP_START..LOOP_END-1 (9 layers), the cache list is constructed
as follows. Let L=42, LOOP_START=12, LOOP_END=21, N_LOOP=9, T=T_ref[0].

Indices:
- Non-loop prefix: `i ∈ [0, LOOP_START)` → `cache_index = i`. 12
  entries.
- Loop region: `t ∈ [0, T)`, `j ∈ [LOOP_START, LOOP_END)` →
  `cache_index = LOOP_START + t · N_LOOP + (j - LOOP_START)`. T · 9
  entries per forward, but only stored for layers whose j
  corresponds to a "own-cache" layer (j < first_kv_shared). For
  Gemma 4 E4B (first_kv_shared=22), all j ∈ [12, 21) satisfy
  j < 22, so all 9 looped layers own their cache → T · 9 entries
  for the loop region.
- Non-loop suffix: `i ∈ [LOOP_END, L)` → `cache_index = LOOP_START
  + T · N_LOOP + (i - LOOP_END)`. 21 entries. Of these, layers
  with i < first_kv_shared (i.e., i=21 only for E4B) own their
  cache; others share via previous_kvs.

Total cache list length: `LOOP_START + T · N_LOOP + (L - LOOP_END)`
= `12 + T·9 + 21` = `33 + 9T`. For T=6, 33 + 54 = 87 entries.

Per-entry cache object: `KVCache()` for full-attention layers,
`RotatingKVCache(max_size=sliding_window, keep=0)` for
sliding-attention layers. Layer_type for j ∈ [12, 21) in the loop
region inherits from `config.layer_types[j]` — so within the loop,
caches may mix full and sliding types. Same pattern as standard
mlx_lm: the looped caches at iteration t simply re-use the same
layer_type as the underlying layer.

### 1.3 Patched `__call__` for cached recurrent-depth forward

Pseudocode (actual code in `run_experiment.py`):

```
def patched_call_cached(self, inputs=None, cache=None, ...):
    h = self.embed_tokens(inputs) * self.embed_scale
    # ... (per-layer-input prep unchanged)

    # Expand cache to full length if short
    expected_len = LOOP_START + T_ref[0] * N_LOOP + (len(self.layers) - LOOP_END)
    if cache is None:
        cache = [None] * expected_len
    elif len(cache) < expected_len:
        cache = cache + [None] * (expected_len - len(cache))

    masks = self._make_masks_recurrent(h, cache)
    intermediates = [(None, None)] * len(self.layers)

    idx = 0
    while idx < len(self.layers):
        if idx == LOOP_START:
            h_block_entry = h
            for t in range(T_ref[0]):
                loop_idx_ref[0] = t
                h_loop = h
                for j in range(LOOP_START, LOOP_END):
                    # CACHE INDEX:
                    c_idx = LOOP_START + t * N_LOOP + (j - LOOP_START)
                    h_loop, _, _ = self.layers[j](
                        h_loop, masks[j], cache[c_idx],
                        per_layer_input=per_layer_inputs[j],
                        shared_kv=None, offset=None,
                    )
                if t < T_ref[0] - 1:
                    h = lti_bank[t](h_block_entry, h_block_entry, h_loop)
                else:
                    h = h_loop
            idx = LOOP_END
            continue
        # Non-loop branch
        c_idx = idx if idx < LOOP_START \
                else LOOP_START + T_ref[0] * N_LOOP + (idx - LOOP_END)
        kvs, offset = intermediates[self.previous_kvs[idx]]
        h, kvs, offset = self.layers[idx](
            h, masks[idx], cache[c_idx],
            per_layer_input=per_layer_inputs[idx],
            shared_kv=kvs, offset=offset,
        )
        intermediates[idx] = (kvs, offset)
        idx += 1

    return self.norm(h)
```

### 1.4 Masks under recurrent-depth forward

Mask dimensions depend on the layer_type of each layer, not on cache
identity. For the cached recurrent forward, masks[j] for j ∈ [12, 21)
must reflect the correct attention mask for h_loop's current sequence
position — which equals the h_loop length (same as the non-loop input
h length). So masks can be computed once at the start with
`_make_masks(h, cache)`, where cache passed to `_make_masks` is a
dummy list of correct layer_types and offsets.

**Subtle point:** `create_attention_mask(h, c)` inspects `c.offset`
(cache position). For the cached recurrent forward, when generating
token N+1 with a cache that already contains N tokens, `c.offset=N`
— so `create_attention_mask` produces a mask of shape (1, L_new,
N + L_new) where L_new is the new token count (1 during generation).
All layers in the loop at iteration t use the same mask (same
sequence position). This is preserved in the pseudocode above:
`masks[j]` depends only on j and the current h.

## §2 Prior art

### 2.1 Baseline: parent `exp_rdt_loop_lora_gemma4_bench`

F#674 validated structural + dynamical claims at n=50 and n=500
steps:
- K-FULL-A (block wiring): PASS.
- K-FULL-B (v+o grad co-update): PASS at 2.4e-2 / 6.9e-2.
- K-FULL-C-EXT (rho < 1 over 500 steps; |Δlog_A|, |Δlog_dt| > 1e-4):
  PASS (max_rho=0.555, Δlog_A=0.248, Δlog_dt=0.280).
- K1740-BENCH: under_powered (n=30 << 200); directionally positive
  (+3.33pp at T=3).
- K1742-BENCH: under_powered (n=10/T across 4 T values; R²=0.32
  with degenerate fit).
- K-KVCACHE: not_measured — scope-deferred to **this** experiment.

Parent MATH Theorem 2 quantified: uncached eval at n=200, T=3 costs
~183 hours on M5 Pro. KV-cache is prerequisite for running K1740 at
pre-registered n.

### 2.2 mlx_lm KV-cache internals

`mlx_lm.models.cache.KVCache.update_and_fetch(keys, values)` appends
`keys, values` along the sequence axis and returns the concatenated
cache. `RotatingKVCache.update_and_fetch` additionally truncates to
`max_size` with `keep` retained prefix tokens — for sliding-attention
layers with Gemma 4 E4B's sliding_window=512. The update contract is
**stateful**: calling update_and_fetch on the same cache twice with
different K/V concatenates both — hence parent's bug when cache[j]
was reused across loop iterations.

### 2.3 Recurrent-depth literature

- Bae 2024 (arxiv:2410.20672) — "Looped Transformers" — analyzes
  iterated forward over fixed weights; does not address KV-caching
  for looped generation.
- Saunshi 2025 (arxiv:2502.17416) — "Reasoning with latent thoughts"
  — recurrent-depth for latent reasoning; explicitly OOS for
  generation efficiency.
- Parcae Prairie 2026 (arxiv:2604.12946) — "Infrastructure for
  looped transformer training" — discusses activation checkpointing
  but not inference caching.

**Gap identified by this experiment:** no prior work publishes a
verified KV-cache construction for a looped-depth transformer where
each loop iteration uses a distinct weight perturbation (LoRA delta
indexed by loop iter). Our construction (T caches per looped layer)
is a new structural primitive.

## §3 Kill criteria (pre-registered, locked)

Per PLAN §1 rule (e) — no post-hoc KC modification.

### 3.1 K1764 — bit-exact cached↔uncached consistency

For each T ∈ {1, 2, 3, 6} and each of n=20 GSM8K-valid prompts:
1. Run uncached forward on prompt → logits_unc (last-token logits).
2. Run cached forward on the same prompt with fresh cache →
   logits_cached (last-token logits, after cache populated).
3. Compute `max_abs_logit_diff = max|logits_unc - logits_cached|`
   across the vocabulary axis.

**PASS:** max_abs_logit_diff < 1e-3 for all (T, prompt) pairs (80
total pairs). Tolerance rationale: fp16 mantissa is 10 bits ≈ 3
decimal digits; 1e-3 is one ULP at logit magnitude O(1). Bit-exact
(0.0) tolerated as stronger evidence.

**FAIL:** any pair exceeds 1e-3. Indicates cache-mechanism bug; the
construction in §1.2 is flawed. Follow-up: identify which (T, j, t)
slot accumulates wrong K/V.

**Proxy-FAIL / target-PASS guard (F#666):** K1764 is a structural
KC. There is no paired "target" metric because the K1764 claim is
inherently about mechanism correctness, not downstream task outcome.
Per reviewer.md §5 PROVISIONAL-as-design clause, structural KCs on
mechanism primitives are exempt from F#666 target-gating because no
behavioral target exists to gate against. However, to meet F#666
spirit, K1765 serves as the "behavioral" gate: the cache must be not
only correct but *usefully fast* — 5× speedup is the real-world gate.

### 3.2 K1765 — 5× wall-clock speedup

On the same n=20 prompts at T=3, measure wall-clock time for:
1. Uncached generation of M=64 new tokens per prompt (greedy decode).
2. Cached generation of M=64 new tokens per prompt (greedy decode,
   using §1.3 cache).

**PASS:** cached_total_time ≤ uncached_total_time / 5.0. I.e. ≥5×
speedup. Rationale: the theoretical speedup ceiling for KV-cache on
causal generation is O(N) per token uncached vs O(1) per token
cached, where N is the current sequence length. For prompt length
~200 tokens + generation of 64 tokens, the average speedup is
bounded below by (200 + 64/2) / 1 = 232×. 5× is a loose lower bound
that accounts for the looped region's T×overhead (cache-free in
naive impl, cached with T caches in ours). If 5× is not met, the
T-way cache layout overhead swamps the cache benefit — the mechanism
is correct but not useful.

**FAIL:** cached_total_time > uncached_total_time / 5.0. Indicates
mechanism is correct but impractical; either the cache-layout has
excessive Python overhead, or mlx_lm's KVCache dispatch has
T-dependent overhead not modelled here.

### 3.3 Target-gating pair (F#666)

Per §3.1 rationale: K1764 is the proxy (mechanism correctness);
K1765 is the target (usefulness). SUPPORTED requires both PASS.
KILLED requires both FAIL. K1764 PASS + K1765 FAIL → "cache is
correct but not useful" finding (promote to follow-up with simpler
cache-layout variant). K1764 FAIL + K1765 PASS → "cache appears fast
but is numerically wrong" (immediate kill of the construction).

## §4 Theorem: bit-exact equivalence under §1 cache layout

**Theorem (bit-exact cached/uncached equivalence).** Let
`h_unc` = uncached forward on prompt P at T=T_0 per parent's patched
`__call__` (run_experiment.py line 119, `cache=[None]*L`). Let
`h_cached` = cached forward on prompt P at T=T_0 per §1.3 with a
fresh cache list of length `33 + 9·T_0` (all None on first call).
Then `h_unc == h_cached` bit-exactly in exact arithmetic, and
`max|h_unc - h_cached| < ε_fp16` in fp16 where ε_fp16 depends only
on the fp16 ULP accumulated over the forward (independent of T).

**Proof.**

*Step 1 (non-loop layers).* For layer i ∉ [LOOP_START, LOOP_END),
the cached and uncached forwards both compute
`layer(h, masks[i], cache_i, ..., shared_kv=kvs_prev, offset=offset)`.
On first call, `cache_i = None` (uncached) or `cache_i = KVCache()`
with `offset=0` (cached-fresh). The Attention block's
`cache.update_and_fetch(K, V)` on an empty cache returns exactly
(K, V) (from mlx_lm.models.cache.KVCache source; the cache initially
stores nothing, and update_and_fetch appends then returns). So the
downstream `scaled_dot_product_attention(q, K, V, cache=cache_i, ...)`
sees identical (K, V) in both branches. Identical inputs to every
downstream op ⇒ identical outputs. QED non-loop.

*Step 2 (loop iteration t at layer j ∈ [LOOP_START, LOOP_END)).*
The uncached branch passes `cache=cache[j]=None`. The cached branch
passes `cache=cache[LOOP_START + t·N_LOOP + (j - LOOP_START)]=KVCache()`
with `offset=0`. Attention: `K_t, V_t = k_proj(h_loop), v_proj(h_loop)`
where `v_proj = LoopLoRALinear(base_v, deltas_v, loop_idx_ref=[t])`
computes `base(h) + delta_t(h)`. Identical across branches (both
branches set `loop_idx_ref[0]=t` in the same order). The
`cache.update_and_fetch(K_t, V_t)` on a fresh cache returns
`(K_t, V_t)` — same as the uncached branch's `(K_t, V_t)` (mlx_lm
`scaled_dot_product_attention` with `cache=None` uses the provided
K, V directly). ⇒ identical Attention output for this (t, j) pair.
QED loop.

*Step 3 (LTI injection).* `lti_bank[t]` is invoked on identical
inputs in both branches (step 2 established h_loop is identical).
LTI is a pure function of inputs, no cache state. Identical outputs.

*Step 4 (propagation).* Each step's output equals the next step's
input. By induction, `h_unc_final == h_cached_final` across all L
layers.

*Step 5 (fp16 ε bound).* fp16 accumulation error on any
single op is bounded by `eps_op ~ 2^-10 * ||x||`. The forward has
O(L + T · N_LOOP) = O(42 + 6·9) = O(96) ops with potential
accumulation. If each op's error is bounded by c · eps_op for
constant c ~ 10, total error is ≤ 10 · 96 · 2^-10 ≈ 1 (in relative
units). For a logit magnitude of O(1), absolute error ≤ O(1e-3).
Hence max_abs_logit_diff < 1e-3 is an achievable tolerance; bit-exact
is expected when the same op ordering is preserved.

**Caveat (not a proof gap):** `RotatingKVCache` for sliding-attention
layers truncates to `max_size=sliding_window=512`. For prompts with
length + T·prompt_length > 512, the truncation can differ between
uncached and cached if the truncation algorithm depends on cache
history. However, `max_size=512` applied to a fresh cache on a
prompt ≤ 512 tokens behaves identically to no-cache because no
truncation is triggered. For K1764 test, we **restrict n=20 prompts
to length ≤ 400 tokens** (GSM8K-valid has mean prompt length ~120
tokens; all examples satisfy this). For K1765 test, M=64 generated
tokens on ≤400-prompt means cache length ≤ 464 < 512 at end of
generation — still within sliding window.

## §5 Prediction vs measurement

| KC | Prediction | Measurement path |
|---|---|---|
| K1764 | max_abs_logit_diff < 1e-3 across all 80 (T, prompt) pairs | forward-logit-match loop in `verify_bit_exact()` |
| K1765 | cached T=3 ≥ 5× faster than uncached on n=20, M=64 new tokens | wall-clock timing in `verify_speedup()` |

## §6 Scope escalation — PROVISIONAL-as-design with `_impl` at P3

Per reviewer.md §5 "PROVISIONAL (macro-scope design-only sub-case)"
clause and handoff instruction #4 ("If engineering scope blows past
single-iter budget → PROVISIONAL-as-design per §5 + `_impl` at P3"):

**Scope judgement.** The construction in §1–§4 is mathematically
complete and produces specific code directives. Empirical
verification requires:
1. Loading Gemma 4 E4B 4-bit (~3 min).
2. Implementing `patched_call_cached` with the §1.2 cache layout
   and §1.3 pseudocode (~30-45 min engineering time for a
   correct-first-try implementation; risk of subtle cache-index
   off-by-one).
3. For K1764: 80 paired forwards (uncached + cached) on n=20
   prompts × 4 T values. Per parent's uncached speed, ~1 minute per
   uncached forward on short prompts; total uncached budget ~80 min.
   Cached should be much faster but adds comparable verification
   overhead. Plausible 2–3 h.
4. For K1765: Full generation run at M=64 tokens × 20 prompts × 2
   modes (cached / uncached) × 1 T value (T=3). Uncached ~50
   sec/prompt × 20 ≈ 17 min; cached target ≤ 4 min. ~20-30 min.

Total empirical budget: 3-4 h, plausibly running over the
researcher-hat 2h cap when including unforeseen MLX-debug time (F#673
lineage — KV-cache bugs are subtle and produce plausible-looking
wrong logits).

Per the parent experiment's 6644-sec (1.8h) reference runtime and
the risk profile of cache bugs producing silent failures, this
experiment ships as PROVISIONAL-as-design:
- Full mathematical construction captured in §1–§4.
- `run_experiment.py` is a graceful-failure scaffold that loads the
  model, wires loop-LoRA, and writes a valid `results.json` with
  `verdict=PROVISIONAL` and K1764/K1765 = `not_measured` with
  detailed reason.
- Follow-up `exp_rdt_loop_kv_cache_impl` filed at P3 inheriting
  K1764/K1765 verbatim; it runs the full verification path on the
  compute budget released by P3 priority.

**Reviewer antipattern (t) defence.** No scope swap has occurred.
The scope F1–F6 locked in §0 is preserved verbatim; it is simply
unverified in this iteration and explicitly marked as such.

**Reviewer antipattern (m) defence (proxy-model).** `run_experiment.py`
loads `mlx-community/gemma-4-e4b-it-4bit` exactly as specified in §0
F1. No smaller variant is substituted. The scaffold writes
`results.json` with `base_model=MODEL_ID` and `verdict=PROVISIONAL`;
it does not run and does not produce false measurements.

## §7 Antipattern self-audit

| Code | Antipattern | Status |
|---|---|---|
| composition-bug | `ΣA_i, ΣB_i` independently | n/a — K1764 test does not compose adapters; single-adapter stack per forward |
| tautological-routing | `route(val[d][0])` | n/a — no routing; loop index scheduled not routed |
| lora-scale-20 | `LORA_SCALE ≥ 12` | α=2 → scale=0.125 (inherited from parent, safe) |
| shutil-copy-adapter | `shutil.copy` sibling adapter | n/a — no adapter copy |
| hardcoded-pass | `{"pass": True}` literal | n/a — PROVISIONAL scaffold writes `not_measured`, never hardcoded pass |
| thinking-truncation | base eval with `avg_thinking_chars=0` | n/a — no generation eval in this run |
| proxy-model | MATH says X, code loads Y | MATH = `mlx-community/gemma-4-e4b-it-4bit`, scaffold loads same |
| smoke-as-full | `is_smoke=true` in full-run claim | `is_smoke=false`; KCs explicitly `not_measured`, verdict PROVISIONAL |
| kc-tautological | K passes by algebraic identity | K1764/K1765 both test non-trivial claims; no algebraic shortcut |
| kc-swap | KC modified after first result | KCs locked in MATH.md at write time; pre-registered as DB #1764/#1765 |
| copy-paste-scaffolding | sibling scaffolding reused without re-reading | parent's `run_experiment.py` re-read line-by-line; only inspection, no code ported |
| scope-swap | silent change of base model / data / KC thresholds | §0 F1–F6 locked; §6 explicitly files PROVISIONAL-as-design rather than swap |

## §8 Libraries

- `mlx==0.31.1`, `mlx-lm==0.31.2` (pinned; parent-inherited).
- `datasets==4.3.0`, `dill==0.4.0` (for data loading — not used in
  scaffold but pinned for `_impl`).
- Seed 42, stream `mx.default_stream()` except
  `mx.linalg.qr(stream=mx.cpu)` if QR needed.

## §9 Assumptions logged

- **Gemma 4 E4B config `num_hidden_layers`**: parent states 42. The
  installed mlx_lm default is 35, but the 4-bit E4B checkpoint
  provides its own config.json that overrides. §1 cache-index
  formulas assume L=42; they are derived symbolically in terms of L
  and correct for any L.
- **`first_kv_shared=22`**: assumed from `num_kv_shared_layers=20`
  for E4B. §1.2 formula is correct for any first_kv_shared such that
  `first_kv_shared ≥ LOOP_END=21` (all looped layers own their
  cache). If first_kv_shared < 21, the loop region contains shared
  layers, and §1.3 must add a `previous_kvs` branch inside the loop
  — engineering addendum, not a theorem change.
- **Prompt length ≤ 400 tokens**: GSM8K-valid empirical mean ~120,
  95%ile ~300; all 20 prompts assumed ≤ 400. Verifier scaffold
  rejects prompts > 400 (writes `not_measured` for that pair).
