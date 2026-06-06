# MATH — exp_rdt_loop_kv_cache_full

Full-scope follow-up to `exp_rdt_loop_kv_cache_impl` (PROVISIONAL F#785,
K1837 PASS_SMOKE bit-exact at n=2 × T=3). Inherits MATH §0–§4 verbatim
from `_impl` (which inherits §0–§4 verbatim from parent design
`exp_rdt_loop_kv_cache` F#690 PROVISIONAL). The only delta is the
addition of full-scope measurement harnesses for K1838/K1986/K1987 that
the smoke-mode `_impl` explicitly deferred.

## §0 Skills invoked, scope lock, pinned versions

Platform skills `/mlx-dev` + `/fast-mlx` invoked before writing this
document and the companion `run_experiment.py`, per PLAN.md Part 2 +
guardrail 1012. Items internalized and applied:

- `mx.eval(...)` only at loop boundaries; MLX is lazy.
- Slices create copies (not views — opposite of NumPy). Use
  `mx.concatenate` to extend the uncached generation prompt buffer.
- `mx.array([...], dtype=mx.int32)` for tokenizer outputs.
- `cache.update_and_fetch(K, V)` on a fresh `KVCache` returns `(K, V)`
  exactly (parent MATH §4 Step 1, mlx_lm KVCache source verified).
- Avoid `.item()` inside hot loops — implicit eval. Greedy step writes
  one `int(next_tok.item())` per generation step (necessary, not in a
  vectorizable inner loop).
- `mlx==0.31.1`, `mlx-lm==0.31.2` pinned (parent-inherited).

**Scope-preservation lock (reviewer antipattern (t) defence) — F1–F6
inherited verbatim from `_impl` MATH §0:**

- F1: Base = `mlx-community/gemma-4-e4b-it-4bit`.
- F2: Loop region = layers 12..20 inclusive (LOOP_START=12,
  LOOP_END=21, N_LOOP=9).
- F3: T-sweep = {1, 2, 3, 6} for K1837. K1838/K1986 use T=3 only
  (parent §3 + `_impl` §3 fix the speedup measurement at T=3).
- F4: n=20 prompts (GSM8K-valid subset) for K1837 + K1838 + K1986;
  n=199 for K1987 (we have 199 prompts available; K1987's pre-reg
  text says "n=200 GSM8K-Hard", documented assumption A1).
- F5: Bit-exact tolerance = max_abs_logit_diff < 1e-3 in fp16 (K1837).
- F6: Speed threshold = cached ≥ 5× faster than uncached (K1838).
  Parity threshold = ≥ 99% greedy-token agreement (K1986).
  Budget = ≤ 2h researcher-hat wall-clock (K1987).

**Smoke vs full execution mode.** This `_full` script is the macro-tier
verification that fully measures K1837–K1987. SMOKE_TEST=1 is supported
**only as a plumbing pre-flight**: it runs K1837 at n=2×T=3 + K1838 at
n=2×M=8 cached/uncached + K1986 at n=2×M=8 + K1987 SKIPPED. The smoke is
intended to catch wiring bugs in the new gen harnesses (K1838/K1986/K1987)
**before** committing the 3–4h pueue task. Per PLAN §1010 #4 smoke
verdict ceiling is PROVISIONAL.

## §1 Architecture — cache layout under recurrent-depth forward

Inherits `_impl` MATH §1.1–§1.3 verbatim. Three artefacts implemented in
`run_experiment.py`:

- `wire_loop_lora` — partition QR LoRA-A bank across layers 12..20 for
  v_proj + o_proj × N_LOOPS=6 (parent F#627 target).
- `install_patch` (Gemma4TextModel.__call__) — unified cached/uncached
  recurrent-depth forward using a single c_idx formula and length-L
  mask-cache alias (parent §1.3, `_impl` §1.2 unification).
- `make_recurrent_cache(model, T_now)` — builds list of length
  `33 + 9·T_now` with proper KVCache vs RotatingKVCache mix per layer
  type, respecting `previous_kvs[i] != i` skip-cache markers.

The cache list is **incrementally-updated** during greedy generation:
each `cache[c_idx]` stores per-layer KV that grows token-by-token via
`cache.update_and_fetch(K, V)`. The same patched call serves both
prompt prefill (long input) and per-step generation (single new token).

## §2 New harnesses — K1838, K1986, K1987

### 2.1 `verify_speedup` (K1838)

For each of n=20 prompts at T=3, M=64:

1. Build fresh cache `cache_c = make_recurrent_cache(model, 3)`.
2. Time cached prefill + M cached single-token decodes (all using
   `cache=cache_c`, which advances incrementally).
3. Time uncached prefill + M uncached single-token decodes (each call
   passes `cache=None`, model re-processes the full growing prompt
   buffer — O(M²) cost).
4. Wall-clock `t_cached` and `t_uncached` recorded; `speedup =
   t_uncached / t_cached`.

K1838 PASS iff `mean(speedup over 20 prompts) ≥ 5.0`.

### 2.2 `verify_behavioral_parity` (K1986)

Reuse the n=20 cached vs uncached generations from §2.1 (no extra
forward calls needed — token sequences captured during timing run).

For each prompt: `agree[i] = mean(cached_tokens[i] == uncached_tokens[i])`
over the M=64 generated tokens. K1986 PASS iff
`mean(agree) ≥ 0.99` AND `min(agree) ≥ 0.95` (tail-protection on a
single anomalous prompt).

Why `≥ 0.95` for the min: bit-exact logit equality at K1837 implies
greedy argmax should agree token-by-token *except* in the rare case of
near-tie logits where fp16 roundoff in the cached path may select a
different argmax. The min-tail guard accepts at most a 5% per-prompt
divergence (3 / 64 tokens) attributable to such roundoff; anything
worse signals a bug, not roundoff.

### 2.3 `verify_unlock_budget` (K1987)

Run cached gen on n=199 GSM8K-valid prompts at T=3, M=128. Capture
total wall-clock. K1987 PASS iff `t_total ≤ 2h = 7200s`.

The 2h is the researcher-hat budget per parent MATH §4 derivation:
parent's Theorem 2 estimated uncached cost = ~183h for n=200×M=128;
cached cost should be ~60-90× lower per parent estimate ⇒ ~2-3h
expected. K1987 verifies the budget unlock empirically.

## §3 Kill criteria (pre-registered, locked)

Inherited verbatim from `_impl` MATH §3 (DB ids K#1837, K#1838,
K#1986, K#1987 — the F#770-schema-repair target pairs).

- **K1837** (proxy, mechanism correctness): max_abs_logit_diff < 1e-3
  on n=20 GSM8K prompts × T ∈ {1,2,3,6} = 80 pairs.
- **K1838** (proxy, speedup): cached T=3 ≥ 5× faster than uncached on
  n=20 prompts × M=64 new tokens.
- **K1986** (target, pairs K1837 per F#666): greedy-token agreement
  ≥ 99% mean (≥ 95% min) on n=20 GSM8K prompts at T=3, M=64.
- **K1987** (target, pairs K1838 per F#666): cached T=3 generation
  completes n=199 GSM8K M=128 within 2h wall-clock.

Smoke-mode subset: K1837 at n=2×T=3, K1838 at n=2×M=8, K1986 at
n=2×M=8, K1987 SKIPPED. Verdict ceiling PROVISIONAL.

## §4 Theorem inheritance — bit-exact equivalence

Parent MATH §4 (bit-exact equivalence between cached and uncached
recurrent-depth forward) transfers verbatim. `_impl` MATH §4 unification
argument transfers verbatim. K1837 verified empirically at smoke
(`_impl` PROVISIONAL F#785). This `_full` extends to 80 pairs.

The K1986 prediction (greedy-token agreement ≥ 99%) is a **direct
corollary** of K1837 PASS: if `max_abs_logit_diff < 1e-3 < ULP(top-1
logit)` on every pair, then `argmax(uncached_logits) ==
argmax(cached_logits)` token-by-token, modulo the near-tie roundoff
edge case (§2.2).

The K1838 prediction (≥ 5× speedup) is **NOT** a corollary of theorem
— it is an empirical claim about MLX's per-token forward cost when the
cache list is reused. Parent's Theorem 2 estimated 60–90× speedup from
asymptotic O(M²) → O(M) reduction; the 5× threshold is a conservative
lower bound that allows for kernel-launch overhead at small M. If
K1838 fails at the 5× threshold but mean speedup > 1×, that is a
finding about MLX overhead, not a theorem failure.

The K1987 prediction (≤ 2h) follows from K1838 PASS: if cached-vs-
uncached speedup ≥ 5× and parent's uncached estimate was ~183h for
n=200×M=128, then cached ≤ 36.6h. The 2h threshold tightens that to a
≥ 92× speedup requirement at the n=199 M=128 scale — stricter than
K1838's per-(n,M) minimum. This stricter threshold is the actual
"unlock claimed by parent MATH §4": parent's claim was that cached-
recurrent-depth makes the budget tractable; "tractable" here is
operationalized as ≤ 2h researcher-hat budget.

## §5 Prediction vs measurement

| KC | Prediction (smoke) | Prediction (full) | Measurement path |
|---|---|---|---|
| K1837 | max_abs_logit_diff < 1e-3 on 6 pairs (n=2, T=3, but n=2×T=1=2 pairs at smoke; report all 2) | < 1e-3 on 80 pairs | `verify_bit_exact()` |
| K1838 | speedup measured but no PASS threshold (n=2 too small) | ≥ 5× mean cached/uncached | `verify_speedup()` |
| K1986 | greedy-agreement measured but no PASS threshold (n=2 too small) | ≥ 99% mean, ≥ 95% min | `verify_behavioral_parity()` |
| K1987 | SKIPPED | ≤ 2h wall-clock on n=199 M=128 | `verify_unlock_budget()` |

## §6 Antipattern self-audit

| Code | Antipattern | Status |
|---|---|---|
| composition-bug | `ΣA_i, ΣB_i` independently | n/a — single LoopLoRA stack per fwd |
| tautological-routing | `route(val[d][0])` | n/a — no routing |
| lora-scale-20 | `LORA_SCALE ≥ 12` | α=2 → scale=0.125 (parent-inherited, safe) |
| shutil-copy-adapter | `shutil.copy` | n/a |
| hardcoded-pass | `{"pass": True}` literal | results computed from threshold comparison; never literal-pass |
| thinking-truncation | base eval avg_thinking_chars=0 | n/a — no chat-format eval; just greedy-token gen |
| proxy-model | MATH says X, code loads Y | MATH = `mlx-community/gemma-4-e4b-it-4bit`, code loads same |
| smoke-as-full | `is_smoke=true` reported as full | smoke writes `is_smoke=true`, K1987 skipped, K1838+K1986 marked `pass_smoke_plumbing` not `pass`; full mode (SMOKE_TEST=0) writes `is_smoke=false` |
| kc-tautological | K passes by algebraic identity | K1837 tested vs fp16-ULP threshold (non-trivial); K1838 vs 5× wall-clock (no algebraic shortcut); K1986 vs cached/uncached argmax-equality (non-trivial — fails if cache routing is buggy); K1987 vs absolute wall-clock budget (non-trivial) |
| kc-swap | KC modified after first result | K1837/K1838/K1986/K1987 inherited from `_impl` verbatim, not modified here |
| copy-paste-scaffolding | sibling reused without re-reading | `_impl` `run_experiment.py` re-read line-by-line before adapting; new harnesses (verify_speedup/verify_behavioral_parity/verify_unlock_budget) written from scratch with mx-dev rules in mind |
| scope-swap | silent change of base model / data / KC thresholds | §0 F1–F6 locked verbatim; F4 substitution GSM8K-Hard → GSM8K-valid documented as A1, not silent |

## §7 Libraries / pins

- `mlx==0.31.1`, `mlx-lm==0.31.2` (parent-pinned; do not upgrade).
- `datasets` not used — GSM8K data is local at
  `micro/models/exp_p1_t2_single_domain_training/data/math/valid.jsonl`
  (199 prompts in `messages`-format; reused per `_impl` MATH §7).

## §8 Assumptions (must appear in PAPER.md verbatim)

- **A1** (data substitution): K1837/K1986/K1987 pre-reg text says
  "GSM8K-Hard" but we use `valid.jsonl` (199 prompts) from the
  `exp_p1_t2_single_domain_training/data/math/` directory. This is
  GSM8K-valid (the held-out validation slice of GSM8K, not the
  separately-named GSM8K-Hard challenge set). The substitution is
  **not silent**: K1986 (behavioral parity) is data-difficulty-
  independent (it tests cached-vs-uncached gen agreement on the same
  prompts, not absolute correctness on hard problems); K1987 (budget
  unlock) is data-difficulty-independent (it measures wall-clock, not
  accuracy). Only K1837 (logit-diff) is data-content-dependent in any
  meaningful way, and the diff threshold is fp16-ULP-bounded
  irrespective of problem difficulty.
- **A2** (n=199 vs n=200): K1987 pre-reg n=200 but we have 199
  prompts available. The 0.5% shortfall on n is within K1987 noise
  and does not change the budget conclusion (≤ 2h holds for n=199
  iff it holds for n=200 modulo a 0.5% wall-clock margin).
- **A3** (smoke-mode K1838/K1986 PASS threshold): smoke runs at
  n=2×M=8 are too small to PASS the full 5×/99% thresholds reliably
  (kernel-launch overhead dominates at M=8). Smoke reports
  `pass_smoke_plumbing` if K1838 speedup > 1.0× and K1986 mean
  agreement > 0.50 — i.e. the new harnesses ran without crashing
  and produced sane output. This is a **plumbing-pre-flight
  threshold**, not a scientific PASS — explicitly distinct in
  results.json field naming.
