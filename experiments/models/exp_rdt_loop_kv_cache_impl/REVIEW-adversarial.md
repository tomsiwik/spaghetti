# REVIEW-adversarial — exp_rdt_loop_kv_cache_impl

**Verdict:** PROVISIONAL (smoke + structural-KC PASS + target-KC `not_measured`)

Canonical pattern per reviewer.md line 62. K1837 PASS_SMOKE (`max_abs_logit_diff = 0.0` bit-exact across n=2 × T=3 = 2 pairs, well under 1e-3 fp16 tol). K1838/K1986/K1987 explicitly `not_measured`, deferred to `exp_rdt_loop_kv_cache_full`.

## Adversarial checklist (18 items)

| # | Item | Status |
|---|---|---|
| a | results.json verdict vs DB status | PASS — both PROVISIONAL |
| b | all_pass vs claim | PASS — `all_pass=false` matches PROVISIONAL ceiling |
| c | PAPER.md verdict line | PASS — "PROVISIONAL (smoke; K1837 PASS bit-exact; ... deferred)" |
| d | is_smoke flag vs claim | PASS — `is_smoke=true` and DB now `provisional`, never `supported` |
| e | KC mutation post-claim | PASS — K1837/K1838/K1986/K1987 inherit verbatim from parent + F#770 schema-repair, no post-run threshold relaxation |
| f | Tautology sniff | PASS — K1837 measures max_abs_logit_diff between two distinct code paths (uncached `cache=None` branch vs cached fresh-KVCache list); PAPER.md sanity section independently inspects cache state at offset=25 (42 non-None entries with `KVCache`/`RotatingKVCache` mix per `previous_kvs[i]==i`), ruling out "cached path silently routes to uncached" failure mode |
| g | K-ID semantics match MATH | PASS — IDs 1837/1838/1986/1987 line up with descriptions in MATH §3, run_experiment.py `kill_criteria` block, and DB record |
| h | LoRA composition bug grep | PASS — no `sum(lora_A`, no `add_weighted_adapter(combination_type="linear"`. Single-adapter stack per forward (`LoopLoRALinear` indexes via `loop_idx_ref[0]`, no summation) |
| i | LORA_SCALE ≥ 12 | PASS — `α=2`, `r=16`, scale = α/r = 0.125 (parent-inherited, well below 12) |
| j | Routing on single sample | N/A — no routing in this experiment |
| k | shutil.copy adapter as new domain | N/A — no adapter copy |
| l | Hardcoded `{"pass": True}` | PASS — `result` field computed (`pass_smoke if k1837_pass else fail_smoke`); literal "pass_smoke" only inside the truthy branch of the threshold comparison |
| m | Proxy-substitution (MATH says X, code loads Y) | PASS — MATH §0 F1 = `mlx-community/gemma-4-e4b-it-4bit`; `MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"` in run_experiment.py:37 |
| m2 | Skill attestation (`/mlx-dev`, `/fast-mlx`) | PASS — MATH §0 explicit cite + PAPER "Pre-flight" cite. Code uses `mx.eval(logits_unc)` per forward, `mx.clear_cache`, `mx.set_memory_limit`, `mx.set_cache_limit`, `mx.array(..., dtype=mx.int32)` for tokenizer outputs, `mx.linalg.qr(stream=mx.cpu)` for partition-QR (matches /mlx-dev guidance) |
| n | Base eval `avg_thinking_chars==0` | N/A — no generation eval in smoke |
| o | Headline n < 15 | TOLERATED — n=2 in smoke is by design; PROVISIONAL ceiling acknowledges underpowered statistics. Full n=20 × T=4 = 80 pairs deferred to `_full` |
| p | Synthetic padding | N/A — 2 hand-coded math prompts, neither degenerate |
| q | Cited-baseline drift | N/A — no external baseline cited |
| r | PAPER.md prediction-vs-measurement table | PASS — table at PAPER.md §"Predictions vs measurements" with all 4 KCs |
| s | Math errors / unsupported claims | PASS — parent §4 lemma cited correctly; theorem-inheritance reasoning (unified §1.2 `patched_call_unified` reduces to two branches that both transit through `cache=None` ⊕ fresh-`KVCache` first-call lemma) is sound |
| t | Target-gated kill (F#666) | N/A — verdict is PROVISIONAL not KILL. Note: F#666 schema-repair (F#770) was already applied to this experiment 2026-04-25 — K1986 paired with K1837, K1987 paired with K1838. Smoke completing K1837 alone with target K1986 `not_measured` is exactly the canonical PROVISIONAL pattern (deferred-not-measured ≠ FAIL) |
| u | Scope-changing fixes | PASS — F1–F6 unchanged from parent. Smoke explicitly relaxes n (2 instead of 20) and T-sweep ({3} instead of {1,2,3,6}); both documented in MATH §0 with `is_smoke=true` and PROVISIONAL verdict ceiling. No silent SFT→LoRA swap, no max_length reduction, no monitoring disable, no base-model swap, no KC drop |

**18/18 PASS or non-blocking.** No blocking fixes; verdict PROVISIONAL canonical.

## Assumptions logged

1. **First-try bit-exact result is genuine, not a routing bug.** Validated by independent cache-state inspection in PAPER.md §"Cache-list verification": at offset=25 after one forward, 42 non-None entries with proper `KVCache`/`RotatingKVCache` mix per `layer_type` and `previous_kvs[i]==i` predicate. The cached forward really did populate KV state and still produced bit-identical logits to the uncached forward, validating parent MATH §4 Step 1 lemma empirically.
2. **Smoke n=2 is sufficient for PROVISIONAL.** Smoke is meant to land *real* measurement (not prove out the full pre-reg). With max_diff=0.0 — bit-exact, not merely under 1e-3 — there is no statistical underpowering concern *for the proxy KC at the smoke scale*. Generalization to n=20 × T={1,2,3,6} is what `_full` will measure; if any pair drifts above 1e-3 in `_full`, that is a `_full` finding, not a smoke result reversal.

## Drain accounting

- Researcher delivered IMPL with bit-exact PASS on first try (no debug-loop, F#673 cache-bug pattern NOT triggered).
- 3rd consecutive HALT-override iter yielding real PROVISIONAL measurement (politeness_impl ~32s, refactor_impl ~48s, this run ~4s end-to-end via pueue task 3).
- This is the 1st HALT-override on a non-Hedgehog axis (kv-cache structural correctness vs behavior-adapter cluster) — broadens HALT-override applicability beyond F#683 cluster.
