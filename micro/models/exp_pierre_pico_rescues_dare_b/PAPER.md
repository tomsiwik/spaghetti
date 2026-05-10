# PAPER.md — Does Pico calibration rescue the failed B-space DARE?

## Verdict: KILLED

Pico calibration does not rescue B-space DARE. The combined method scored 47.3% avg — 8pp *worse* than bare DARE-B (55.3%) and 17.3pp below Fisher-Rao (64.7%). MedQA collapsed further from 30% to 20%. The concentrated-B hypothesis is rejected: the failure is structural to LoRA's multiplicative interaction under random dropout.

## Prediction vs Measurement

| Metric | Prediction | Measured | Δ |
|--------|-----------|----------|---|
| K1: Beat FR by ≥3pp | Pico+DARE-B ≥ 67.7% | 47.3% | **FAIL** −17.3pp vs FR |
| K2: Recover ≥9pp from bare DARE-B (≥64.3%) | ≥ 64.3% | 47.3% | **FAIL** −8.0pp regression |
| K3: Preprocess ≤ 7s | ≤ 7s | 1.2s | **PASS** |
| K4: MedQA ≥ 50% | ≥ 50% | 20% | **FAIL** worse than prior 30% |

## Method Comparison (N=50 per benchmark)

| Method | GSM8K | HumanEval | MedQA | Avg |
|--------|-------|-----------|-------|-----|
| Single best (per-bench oracle) | 66.0 | 78.0 | 42.0 | 62.0 |
| Fisher-Rao (Pierre default) | 68.0 | 68.0 | 58.0 | 64.7 |
| **Pico+DARE-B (under test)** | **62.0** | **60.0** | **20.0** | **47.3** |
| DARE full-delta (upper bound) | 72.0 | 80.0 | 62.0 | 71.3 |

## Failure Mode Decomposition

The MATH.md posed two competing diagnoses for DARE-B's collapse:
- **(a) Multiplicative interaction**: DARE's expectation-preservation guarantee holds for additive deltas, not LoRA's A@B factored form. Random B-dropout + rescale doesn't preserve E[A@B].
- **(b) Concentrated B**: Random 90% dropout zeros key structured entries in B, especially harming specialized adapters (medical).

**Result: (a) is dominant, (b) is not even secondary.**

Pico explicitly addresses concentrated B by dampening over-shared output directions. If (b) were the primary cause, Pico calibration should have recovered most of the collapse. Instead:
- Pico+DARE-B (47.3%) is *worse* than bare DARE-B (55.3%) by 8pp
- MedQA (the most diagnostic metric for concentrated-B failure) got *worse* (20% vs 30%)

Pico's SVD calibration reshapes B's variance structure, but DARE's random dropout then destroys this calibration. The two methods are anti-synergistic: Pico concentrates signal into fewer directions to reduce interference, DARE randomly drops entries regardless of their importance. The calibrated B has *more* concentrated signal (by design), making random dropout *more* destructive.

## Interpretation

This is a clean interpretability result. B-space DARE's failure is not about B-matrix concentration — it's about LoRA's factored structure being incompatible with element-wise random dropout. Any method that operates on B alone and uses random masking will fail for the same reason: you cannot preserve E[A@B] by only controlling B.

Viable B-space composition must use structure-aware methods (TIES sign consensus, Pico SVD calibration without dropout) rather than random masking. Full-delta DARE works because it operates on the fused A@B product where dropout preserves expectations correctly.

## References

- Pico (arxiv 2604.16826)
- DARE (arxiv 2311.03099)
- Prior: exp_pierre_dare_b_vs_fisher_rao (DARE-B → 55.3%, MedQA 30%)
- Prior: exp_pierre_pico_calibration (Pico alone: supported)
