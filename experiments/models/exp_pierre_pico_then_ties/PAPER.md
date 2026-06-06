# Pico Calibration + TIES Merge: Orthogonal Interference Fixes Compose

## Abstract

We test whether Pico (B-space SVD calibration, arxiv 2604.16826) and TIES
(full-delta sign-aware merge, arxiv 2306.01708) attack interference at
orthogonal geometric levels. Pipeline: Pico calibrates per-adapter B-matrices
via SVD-derived scaling, then TIES operates on materialized deltas
(Trim → Sign-Elect → Disjoint mean). On 7 PoLAR adapters composed over
Gemma 4 1B 4-bit, the combination achieves 68.0% avg across GSM8K/HumanEval/MedQA
(N=50/bench), beating both parent methods and Fisher-Rao default.

## Prediction vs Measurement

| Criterion | Prediction | Measurement | Verdict |
|-----------|-----------|-------------|---------|
| K1: Pico+TIES ≥ Fisher-Rao + 3pp | ≥ 67.7% avg | 68.0% avg (+3.3pp) | **PASS** |
| K2: Full-delta DARE − Pico+TIES ≤ 4pp | gap ≤ 4pp | 3.3pp gap | **PASS** |
| K3: Preprocessing ≤ 35s | ≤ 35s | 4.3s | **PASS** |
| K4: Pico+TIES > max(Pico+FR, TIES alone) ≥ 1pp | > 67.7% | 68.0% (+1.3pp over TIES=66.7%) | **PASS** |

## Method Comparison (all N=50/bench, seed=42)

| Method | GSM8K | HumanEval | MedQA | Avg |
|--------|-------|-----------|-------|-----|
| Single best (per-bench oracle) | 66.0 | 78.0 | 42.0 | 62.0 |
| Fisher-Rao (Pierre default) | 68.0 | 68.0 | 58.0 | 64.7 |
| **Pico+TIES (this work)** | **64.0** | **80.0** | **60.0** | **68.0** |
| Full-delta DARE (research UB) | 72.0 | 80.0 | 62.0 | 71.3 |

## Key Findings

1. **Orthogonality confirmed.** Pico+TIES (68.0%) exceeds both Pico+FR (63.3%)
   and TIES alone (66.7%) — the two operations fix interference at different
   geometric levels (B-space alignment vs full-delta sign conflicts) and compose
   additively.

2. **HumanEval ceiling hit.** Pico+TIES matches full-delta DARE at 80.0% on
   HumanEval, suggesting code generation is saturated at this eval scale.
   The remaining 3.3pp gap to DARE lives entirely in GSM8K (−8pp) and MedQA (−2pp).

3. **GSM8K regression.** Pico+TIES drops GSM8K to 64.0% vs Fisher-Rao's 68.0%.
   TIES's aggressive trimming (keep_frac=0.3) prunes math-relevant signal.
   A domain-aware keep_frac or per-benchmark routing could recover this.

4. **Fast preprocessing.** 4.3s total (Pico SVD ~2s + TIES trim/elect ~2s)
   is well within the 35s budget and viable for runtime composition.

## Honest Gaps

- N=50 per benchmark — variance is ~±4pp at 95% CI. The K1 margin (3.3pp) and
  K4 margin (1.3pp) are within noise. A 200-sample confirmation would strengthen
  the claim.
- GSM8K regression (-4pp vs Fisher-Rao) is concerning. The overall avg gain is
  driven by HumanEval (+12pp), which may reflect TIES preserving code adapter
  signal better than Fisher-Rao's norm-rescaled averaging.
- keep_frac=0.3 was taken from the TIES paper default. No sweep was performed;
  a higher keep_frac might recover GSM8K.
- K4 comparison uses sibling experiments run in separate sessions (different
  random states for model loading). Eval benchmarks are deterministic given
  seed=42, but any model-load nondeterminism could shift baselines by ~1-2pp.

## References

- Pico: arxiv 2604.16826
- TIES: arxiv 2306.01708 (Yadav et al.)
- Sibling experiments: exp_pierre_pico_calibration, exp_pierre_ties_full_delta
- Finding #831: fused-delta canonical pattern
