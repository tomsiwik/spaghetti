# MATH: Per-sample routing across full validation set — does pierre_ppl = single_ppl?

## Context
Finding #553 proved single-sample routing produces `pierre_ppl ≡ single_ppl`
**deterministically** when routing accuracy is high. This follow-up asks whether
**per-sample routing** on the full validation set breaks that tautology.

## Theorem 1 (Combinatorial routing bound)
Let:
- `D` = number of domains (= 5), `N` = validation samples per domain (= 50).
- `p` = per-sample router accuracy, assumed i.i.d. across samples.
- `single_ppl[d]` = PPL of val[d] under adapter `d` applied to *every* sample of val[d].
- `pierre_ppl[d]` = PPL of val[d] under adapter `route(x_i)` applied *per-sample*.

Define the event `E = {route(x_i) == d for all x_i in val[d], for all d}`. Then:

    P(E) = p^(N·D)

Conditional on `E`, `pierre_ppl[d] = single_ppl[d]` for all d by construction.
Conditional on `¬E`, at least one sample routes to a non-oracle adapter and the
per-sample PPL mean differs (unless the wrong adapter happens to produce the
identical per-token log-prob — a measure-zero coincidence for continuous losses).

So:  **P(pierre_ppl[d] ≡ single_ppl[d] for all d)  ≤  p^(N·D)  <  1** for p ∈ (0,1), N≥1, D≥1.

## Corollary (K1549 satisfaction by derivation)
At `p = 0.85`:  `p^(N·D) = 0.85^250 ≈ 2.1·10⁻¹⁸`.
At `p = 0.99`:  `p^(N·D) = 0.99^250 ≈ 0.082`.
At `p = 0.996`: `p^(N·D) = 0.996^250 ≈ 0.368`.

For any `p ≤ 0.99`, the probability that per-sample routing reproduces
`pierre_ppl ≡ single_ppl` is below 10%. The identity observed in
`exp_pierre_unified_pipeline/results.json` (all 5 domains to 3 decimal places)
is therefore **not** explainable by per-sample routing; it is the artifact of
single-sample routing at `val[d][0]` (Finding #553).

## Predictions (if experiment were runnable)
- At measured `p = 0.996`, per-sample routing should produce a small but
  nonzero gap: `|pierre_ppl[d] − single_ppl[d]| > 0` for at least 1 domain
  with ≈63% probability per run.
- Expected magnitude: for the ~1 in 250 misrouted samples with average
  base−single gap ~3 PPL (from `results.json`), a 1/N=1/50 contribution shifts
  mean PPL by ~0.06 — measurable but small.

## Kill Criteria
K1549 (original): *"At 85–99% per-sample routing accuracy, pierre_ppl and
single_ppl differ (not identical by construction)."*

**Theorem 1 settles K1549 by derivation** without measurement: identity requires
probability-zero coincidence; it is not forced by construction under per-sample
routing.

## Measurement plan (blocked)
Required inputs for empirical verification:
1. Trained adapter weights for all 5 domains (math, bash, python, sql, medical).
2. Per-sample forward pass through each adapter for each sample in `val[d]`.
3. Independent measurement of router accuracy `p` under per-sample (not
   single-sample) dispatch.

## Pre-flight check (mandatory)
- `ls adapters/{math,bash,python,sql,medical}/adapters.safetensors` → **0 of 5 present** (stubs only).
- `find . -name adapters.safetensors -path '*/pierre*/*'` → 0 results.
- Upstream Pierre siblings (`pierre_unified_pipeline`, `pierre_v3_sft_n5`,
  `pierre_v5_ternary_lora`, `pierre_v6_precomputed_concat`) → all `status=killed`,
  no adapter weights saved.

## Antipattern self-check
- antipattern-017 (stub-adapter cascade, consumer side): **TRIGGERS**.
  5 of 5 adapter directories in `adapters/` contain only `adapter_config.json`.
- antipattern-020 (cascade-dependent design): **TRIGGERS**. Upstream Pierre
  siblings are all killed.
- antipattern-003 (LORA_SCALE): N/A.
- antipattern-008 (thinking-truncation): N/A.
- KC-swap: clean (MATH.md single commit).

## References
- Finding #553: Tautological-routing antipattern (supported, 2026-04-17).
- `micro/models/pierre_unified_pipeline/results.json` — identity evidence.
- `.ralph/agent/memories.md` antipattern-017, antipattern-020.
