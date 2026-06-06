# Adversarial Review: exp_bench_gpqa_thinking

## Verdict: PROCEED (KILLED)

## Review

Clean kill. Two of three kill criteria failed catastrophically, and the data is consistent.

### Checklist

| Check | Result |
|-------|--------|
| Prediction-vs-measurement table | PASS — present, 7 rows |
| Kill criteria match evidence | PASS — K1458/K1459 FAIL, K1460 PASS, all match results.json |
| Finding status appropriate | PASS — KILLED is correct for 2/3 hard failures |
| Data integrity | PASS — 198 questions, per-domain totals sum correctly (30+26+5=61) |
| No fabrication | PASS — results.json aligns with all PAPER.md claims |

### Strengths

1. **Decisive result.** -1.0pp thinking boost (vs predicted +26pp) is unambiguous. No
   borderline interpretation needed.
2. **Good impossibility structure.** Quantization error compounding over N reasoning steps
   is a mechanistically sound explanation. Each thinking step re-enters quantized weights,
   accumulating noise.
3. **Useful architectural insight.** Establishes a quantization ceiling (~31%) on GPQA Diamond
   under 4-bit, independent of inference strategy. This constrains future benchmark expectations.

### Minor Issues (Non-blocking)

1. **"Theorem" is empirical extrapolation, not a theorem.** MATH.md calls the constant
   Δ_think claim a "Theorem (Empirical Prediction)" — this is honestly labeled, so acceptable
   for a benchmark experiment, but it's a correlation observation, not a derived guarantee.

2. **Error compounding model is informal.** The "error ~ N × ε" claim assumes linear
   accumulation, but autoregressive generation could produce superlinear compounding.
   The directional conclusion is correct either way — just noting the bound is loose.

3. **MMLU-Pro thinking not yet verified.** PAPER.md references exp_bench_mmlu_pro_thinking
   as a cross-check. If that also shows zero thinking boost, the quantization ceiling
   claim strengthens from "observed on GPQA" to "general 4-bit property."

### Conclusion

The experiment cleanly establishes that 4-bit quantization eliminates the thinking-mode
benefit on GPQA Diamond. The model generates 1.2M chars of syntactically valid but
semantically broken reasoning. KILLED status is correct.
