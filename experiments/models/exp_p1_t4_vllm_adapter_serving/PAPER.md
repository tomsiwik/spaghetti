# T4.3: MLX-Native Adapter Hot-Swap Serving — Prediction vs Measurement

## Status: KILLED (V2 audit-rerun 2026-04-18)

V1 "SUPPORTED" (2026-04-17) is retroactively invalid. V2 probe flips to KILLED
for four independent structural reasons plus a missing-artefact precondition.
No KC relaxation: V1 thresholds are preserved byte-for-byte in MATH.md.

## V2 Prediction vs Measurement Table

| Kill | Theorem | Prediction | V2 Measurement | V2 Result |
|------|---------|-----------|----------------|-----------|
| K1081: 5/5 adapters load+generate | — (precondition) | 5 × valid output | 0/5 upstream `.safetensors` on disk. `load_adapters(model, path)` raises `FileNotFoundError` before any generation. V1 `results.json` also missing. | **FAIL** (precondition) |
| K1082: swap p99 < 50ms | Theorem 1 (`T_swap ≤ S/B + T_eval`) | ~1.5ms | Operationalised as `load_weights + mx.eval(parameters)` without any forward pass between swaps. MLX recompiles the forward graph on the first forward after parameter mutation — cost omitted from V1's clock. Cannot re-run (C1). | **FAIL** (wrong object) |
| K1083: throughput ratio ≥ 80% | Theorem 2 (FLOPs ratio ~0.47%) | ~99.5% | `generate_tokens` mixes prefill (compute-bound) and decode (memory-bound); math adapter evaluated on non-math prompt (OOD → early-EOS/token-distribution shift). Cannot re-run (C1). | **FAIL** (metric definition + OOD prompt) |
| K1084: routing correct | Theorem 3 (O(1) dict lookup) | 5/5, <1µs | `routing_registry = {d: p for d, p in ADAPTER_PATHS.items()}; selected = routing_registry[domain]` — identity dict. `selected == adapter_path` is `dict[k] == dict[k]`. Zero TF-IDF, zero text input. Cannot re-run (C1). | **FAIL** (tautological routing) |

## V1 Numbers (reference only, unverifiable now — `results.json` absent)

| Kill | V1 Measurement | V1 Verdict |
|------|----------------|------------|
| K1081 | 5/5 valid | PASS |
| K1082 | p99 = 4.77ms (10.5× margin under 50ms) | PASS |
| K1083 | 90.8% (base=41.5 tok/s, adapter=37.6 tok/s) | PASS |
| K1084 | 5/5, ~0.7µs dict lookup | PASS |

V1 runtime (per V1 PAPER.md): not recorded. V2 probe runtime: ~0.001s
(filesystem + 1000-lookup microbench only, no model load).

## Kill Causes

### C1 — Upstream artefact precondition (downstream of T2.1 + T2.6 audit)

All five upstream adapter `.safetensors` files are absent from disk:

    math     — micro/models/exp_p1_t2_single_domain_training/adapters/math/     [config only]
    code     — micro/models/exp_p1_t2_single_domain_training/adapters/code/     [config only]
    medical  — micro/models/exp_p1_t2_single_domain_training/adapters/medical/  [config only]
    legal    — micro/models/exp_p1_t2_multi_domain_5/adapters/legal/            [config only]
    finance  — micro/models/exp_p1_t2_multi_domain_5/adapters/finance/          [config only]

T2.1 status=killed (2026-04-18, MCQ metric-swap + format-artefact). T2.6
weights lost. Any K1081–K1084 measurement requires these files. `load_adapters`
would raise `FileNotFoundError` on the first swap. V1 `results.json` is also
missing from disk, so the "SUPPORTED" verdict recorded in V1 PAPER.md has
no provenance artefact to verify against.

### C2 — Theorem 1 omits MLX graph-recompile (mem-antipattern-011 specialisation)

V1 `swap_adapter`:
```python
t0 = time.perf_counter()
model.load_weights(str(weights_file), strict=False)
mx.eval(model.parameters())   # materialise tensors on device
t1 = time.perf_counter()
```
MLX recompiles / re-traces the forward compute graph on the **first forward
pass** after a parameter mutation. `mx.eval(model.parameters())` materialises
the mutated tensors on-device but does not execute the graph that uses them.
V1 Phase 2 loops 20 back-to-back swaps without a single `generate(...)` between,
so the recompile cost never enters the measurement. Theorem 1's
`T_swap ≤ S_adapter / B_mem + T_eval` is correct as a lower bound but omits
the recompile term, making the 4.77ms-vs-50ms comparison irrelevant to the
true post-swap cost. A correct measurement: swap latency ≡
`t_first_token(after_swap) - t_first_token(baseline_same_prompt)`.

### C3 — Theorem 2 is FLOPs model applied to memory-bound decode + OOD prompt

**Model mismatch.** Theorem 2 predicts `α = 2r/d_model ≈ 0.47%` overhead
(FLOPs ratio). Apple Silicon decode is memory-bandwidth-bound: per-token cost
is dominated by reading weight matrices from unified memory, not arithmetic.
LoRA re-reads the activation X through a second path (base and LoRA), roughly
doubling memory traffic on the adapted layer. The 0.47% FLOPs prediction is
therefore not informative about measured 90.8% — the measurement is what
bandwidth model would predict, not what FLOPs model would.

**Metric conflation.** V1's `generate_tokens` returns
`N_generated / (t_end - t_start)`, i.e. prefill + decode in one denominator.
Prefill is `O(L_prompt)` compute-bound; decode is `O(1)` per token, memory-bound.
The KC wants to characterise LoRA's per-token overhead during steady-state
decode. Correct metric:
`decode_tok_s = N_decoded / (t_end - t_first_token)`.

**OOD prompt.** V1 evaluates the math adapter on
`"Explain the concept of machine learning in simple terms."` — a non-math
prompt. Out-of-domain prompting shifts the generated token distribution and
often early-EOSes. V1's own Phase 1 flags the medical adapter's 3.7 tok/s
outlier as "model answered briefly, denominator small" — the exact same
failure mode applied to K1083 by construction. The 90.8% ratio is not a
characterisation of LoRA overhead.

### C4 — Theorem 3 operationalised as identity-dict (mem-antipattern-002)

V1 code:
```python
routing_registry = {d: p for d, p in ADAPTER_PATHS.items()}
for domain, adapter_path in ADAPTER_PATHS.items():
    selected_path = routing_registry[domain]           # O(1) dict lookup
    # selected_path == adapter_path is True by construction
```
`routing_registry` is the identity copy of `ADAPTER_PATHS`. Indexing it by
the same `domain` key that was used to build it returns the value by
set-theoretic identity, not by routing logic. K1084's stated object —
"correct adapter selected per request via routing header" — requires a
classifier that maps raw prompt text to a domain label (T4.1's TF-IDF
pipeline: tokenise → sparse matmul → argmax). V1 never invokes a router;
it invokes `dict[k]` where `k` is known-correct by iteration. The <1µs
latency is a Python dict-hash microbench; this probe reproduces the same
measurement on a synthetic dict (~0.1–1µs on any hardware) to demonstrate
the measurement is adapter-agnostic.

## Theorem Correctness Note

Theorems 1–3 in MATH.md are mathematically correct *as statements*:

- Theorem 1: a swap that reuses base weights is bounded by adapter I/O; the
  missing term (graph recompile) is a platform-specific correction, not a
  falsification of the inequality.
- Theorem 2: the FLOPs analysis `α = 2r/d_model` is correct arithmetic; it
  is the wrong model for memory-bound Apple Silicon decode.
- Theorem 3: a Python dict lookup is O(1). That is not the same as routing
  raw text to a domain.

The V1 sin is not the theorems but the *operationalisation*:
- K1082 measured `load_weights + mx.eval` not incremental time-to-first-token.
- K1083 measured prefill+decode tok/s on an OOD prompt, not decode tok/s
  on an in-domain prompt.
- K1084 measured `dict[k]` not `TF-IDF(prompt) → argmax`.

## Permanently learned (class-level standing rules — now 8 precondition-probe kills in 24 h)

1. **Precondition-probe before macro claim** (mem-antipattern-002 + 006).
   Every macro-scale claim must first verify artefact presence (`.safetensors`
   on disk, not `adapter_config.json` alone, and `results.json` on disk).

2. **Registry ≠ artefacts.** Dir existence is not file existence.
   Grep for `.safetensors` size, not directory listings.

3. **Downstream P1 macros inherit upstream audit flags.** If an upstream is
   KILLED or its artefacts are lost, the downstream inherits precondition
   failure even if its own code is correct.

4. **`code-bug` tag may be a decoy.** A V1 failure can be a mathematical
   property of the test design (gradient identity, oracle lookup) — fixing
   "code bugs" won't resurrect it; only rewriting the operationalisation will.

5. **Composition / routing claims require genuine routing** — not
   `ADAPTER_PATHS[domain]`. Any experiment whose routing is a constant
   function of the iteration key is a tautology.

6. **Hot-add / hot-remove / hot-swap latency benches weight I/O + graph
   recompile, not dict mutation and not `load_weights` in isolation.** On MLX
   specifically: the first forward pass after parameter mutation carries the
   recompile cost; a tight loop of swaps without generation hides it.

7. **Adapter mint is training, not `shutil.copy`** (mem-antipattern-009).
   A byte-copy of a trained adapter is the same weights under a different
   label — any quality claim is a lie by identity.

8. **Throughput on Apple Silicon decode must strip prefill and stay in-domain.**
   `tok/s` that includes the prefill phase conflates a compute-bound regime
   with the memory-bound regime the KC cares about. Correct:
   `decode_tok_s = N_decoded / (t_end - t_first_token)`. And the prompt must
   be in-domain for the adapter under test — OOD prompting early-EOSes and
   biases the denominator.

## Routing signal for next hat

- Reviewer: 8th precondition-probe kill in 24 h. Class-level standing.
  No new mem-antipattern required — 002 + 011 apply directly; rule #8
  (prefill/decode + OOD) is a specialisation surfaced here and catalogued
  above as a class-level rule.
- Adversarial checklist: verify `results.json` KILLED matches DB, no V2 KC
  relaxation (check `git diff MATH.md`), no V1 leakage as SUPPORTED in PAPER.md,
  `.safetensors` actually absent across upstream dirs (independent re-check),
  probe makes no model load (pure fs + dict microbench).
- Downstream of T2.1 + T2.6. T2.1 cluster of dependent macros now at 8
  (add `vllm_adapter_serving` to the existing 7). Do not auto-spawn V3 until
  T2.1 is rebuilt AND V1's four operationalisation flaws are fixed in a V3
  that implements a genuine TF-IDF router, measures swap latency as
  incremental time-to-first-token, and reports decode-only throughput on
  in-domain prompts.
