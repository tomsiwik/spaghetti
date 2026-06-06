# PAPER — exp_rdt_loop_kv_cache_impl

**Verdict:** PROVISIONAL (smoke; K1837 PASS bit-exact; K1838/K1986/K1987
deferred to `exp_rdt_loop_kv_cache_full`).

## Pre-flight

- Skills `/mlx-dev` + `/fast-mlx` invoked before writing this experiment's
  MATH.md and `run_experiment.py` (PLAN.md guardrail 1012).
- Inherited MATH from parent `exp_rdt_loop_kv_cache` MATH §0–§4 verbatim;
  MATH.md §1.2 unifies the two-call signature into a single
  `patched_call_unified` (IMPL-grade tightening; theorem unchanged).

## Predictions vs measurements

| KC | Prediction | Measurement (smoke) | Verdict |
|---|---|---|---|
| K1837 | max_abs_logit_diff < 1e-3 across all (T, prompt) pairs | 0.000000e+00 across 2/2 pairs (n=2 × T=3) | PASS (bit-exact, well under 1e-3 tol) |
| K1838 | cached T=3 ≥ 5× faster than uncached on n=20 × M=64 | not_measured (smoke skip) | deferred to `_full` |
| K1986 | greedy-token agreement ≥ 99% on n=20 × T=3 × M=64 | not_measured (smoke skip) | deferred to `_full` |
| K1987 | cached T=3 n=200 × M=128 ≤ 2h wall-clock | not_measured (smoke skip) | deferred to `_full` |

## Smoke configuration

- Mode: `SMOKE_TEST=1`.
- Prompts: 2 hand-coded short strings (≤ 50 tokens) — `"What is 12 plus 7?"`
  (n_tokens=25 after chat-template) and `"If a train travels 60 km in 2
  hours, what is its average speed?"` (n_tokens=35).
- T: 3 (single value, not full sweep). 2 pairs total.
- Wall-clock: 1.8s end-to-end (4 forwards: 2 uncached + 2 cached).

## Cache-list verification (sanity)

Independent inspection (separate from the K1837 run) confirms the cached
branch actually populates KV state:
- Cache list length = 60 = `33 + 9·3` (matches MATH.md §1.1 formula
  for T=3, L=42, LOOP_START=12, LOOP_END=21).
- 42 non-None entries (12 prefix + 27 loop-region + 3 suffix-owners
  21,22,23). 18 None entries for shared-kv suffix layers 24..41
  (per `previous_kvs[i] != i`).
- All non-None caches at offset=25 (prompt length) after one forward.
- Layer-type mix correct: `KVCache` for full_attention (5, 11, 17 in loop
  region across T=3, and 23, 29, 35, 41 in suffix), `RotatingKVCache`
  for sliding_attention.

This rules out the failure mode "cached path silently routes to uncached
code" — the cached forward really did use the cache and still produced
bit-identical logits to the uncached forward, validating parent MATH §4
Step 1 lemma empirically.

## Why PROVISIONAL not SUPPORTED

PLAN §1010 #4: smoke completes as PROVISIONAL, never SUPPORTED.
Additionally, F#666 target-gating: K1837 (proxy: mechanism correctness)
must be paired with K1986 (target: behavioral parity) for SUPPORTED, and
K1838 (proxy: speedup) with K1987 (target: budget unlock). Both target
KCs are `not_measured` → verdict ceiling PROVISIONAL even at K1837 PASS.

## Why PROVISIONAL not KILLED

Only proxy KC measured (K1837) PASSED at fp16 bit-exact; no FAIL signal.
Target KCs are explicitly `not_measured` (deferred-not-measured ≠ FAIL,
per F#666 carve-out for forthcoming target-pair work). PLAN §1010 #4
allows PROVISIONAL with measured-PASS proxy + deferred target.

## Antipattern self-audit (mid-run)

| Code | Antipattern | Status |
|---|---|---|
| smoke-as-full | `is_smoke=true` declared in `results.json` and labeled here as smoke; no claim of full coverage. PASS |
| proxy-model | MATH §0 F1 = `mlx-community/gemma-4-e4b-it-4bit`; code loads same. PASS |
| scope-swap | F1–F6 unchanged; smoke explicitly relaxes n & T-sweep with documented reason; PROVISIONAL ceiling. PASS |
| hardcoded-pass | `result` field computed from `max_abs_logit_diff < tol`; not literal `"pass"`. PASS |
| copy-paste-scaffolding | parent bench `run_experiment.py` re-read; classes (LTIInjection, LoRADelta, LoopLoRALinear, partition_qr_lora_A, wire_loop_lora) ported with attention. PASS |
| kc-tautological | K1837 tests max diff against fp16 ULP-bounded threshold (1e-3); non-trivial because cache mechanism could be wrong. The result was bit-exact (0.0) — non-tautological because a wrong cache layout (e.g., parent's bench-time bug accumulating T·(K,V) per token) would have produced large differences. PASS |
| kc-swap | K1837/K1838/K1986/K1987 inherit verbatim from DB IDs #1837/#1838/#1986/#1987; thresholds and semantics unchanged. PASS |
| target-gated kill (F#666) | n/a — verdict is PROVISIONAL not KILLED. |

## Hand-off

Reviewer next iter: REVIEW-adversarial.md template applies (PROVISIONAL
smoke verdict + structural-KC PASS + target-KC not_measured). Expected
verdict-pre-flight items: (s) smoke-as-full PASS, (t) scope-preservation
PASS, (m2) skill-attestation PASS, (h) hardcoded-pass PASS,
(target-gated kill) N/A.

`exp_rdt_loop_kv_cache_full` (P=2 macro) follow-on inherits MATH.md and
runs the full n=20 × T={1,2,3,6} K1837, n=20 × M=64 K1838 (5× speedup),
n=20 × T=3 × M=64 K1986 (greedy-token agreement), and n=200 × M=128
K1987 (2h budget unlock) on a 3-4h pueue task. No ANTHROPIC_API_KEY
required (all KCs are local measurements).

## Operational notes (for analyst)

- Pueue task 3 ran in 4 sec end-to-end. Smoke pattern proven for 3rd
  consecutive iter (politeness_impl ~4 sec wiring + 32s gen, refactor_impl
  48s, this run 1.8s). HALT-override pattern continues to work.
- F#673 cache-bug debug-loop signal NOT triggered: this experiment
  validated bit-exact equivalence on first try. No subtle MLX/cache bug
  manifested in the smoke window.
- F#770 schema-repair (target-pair completeness): this `_impl` already
  has K1986+K1987 paired with K1837+K1838 in DB; smoke verdict ceiling
  PROVISIONAL respects the F#666+F#770 contract.
