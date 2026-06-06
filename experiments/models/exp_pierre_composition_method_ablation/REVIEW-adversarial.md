# Adversarial Review: exp_pierre_composition_method_ablation

## Verdict: KILL CONFIRMED

Finding #830. K2121 FAIL (M3 avg 61.1 < M1 64.4), K2123 FAIL (ρ=0.009, p=0.93).

## Blocking Checklist

| # | Check | Pass? |
|---|-------|-------|
| 1 | MATH.md exists with hypothesis + predictions | Yes |
| 2 | Kill criteria pre-registered before results | Yes |
| 3 | At least one TARGET-METRIC KC (task accuracy) | Yes (K2121) |
| 4 | results.json machine-readable, matches PAPER | Yes |
| 5 | Composition uses Σ(B_i @ A_i) not (ΣB)@(ΣA) | Yes — `(a @ b) * w * SCALE` per adapter |
| 6 | No `__call__` override on PoLAR modules | Yes — uses `_FusedDeltaLinear(nn.Module)` |
| 7 | LORA_SCALE ≤ 8 | Yes (imports SCALE from polar_train) |
| 8 | Same eval slice across all methods | Yes — `build_eval_tuples()` shared |
| 9 | N ≥ 30 per benchmark (not smoke) | Yes (N_BENCH_EVAL=30) |
| 10 | PAPER.md prediction-vs-measurement table | Yes |
| 11 | Relative ordering prediction correct | Yes (M1 > M2 > M3 predicted, confirmed) |
| 12 | KC verdicts internally consistent | Yes |
| 13 | No hallucinated metrics | Yes — results.json aligns with PAPER |

## Observations

1. **Absolute predictions too high** — HumanEval predicted ~90 but measured 70.0. PAPER correctly attributes this to different eval slice vs ties_dare. Not a protocol failure.

2. **Gate peakedness validates kill** — ρ=0.009 with p=0.93 is effectively zero calibration. The gate classifies domains but provides no signal about prompt difficulty.

3. **NaN warnings noted but not blocking** — adapter weight overflow is a known issue (Finding #829). Does not invalidate the relative comparison since it affects all methods equally.

4. **Pierre v1 recommendation sound** — uniform 1/N via FusedDeltaLinear is the correct conclusion. Gate repurposed for sparse selection (which adapters) not continuous weighting (how much).

## Status

Experiment completed as KILLED. All P1+P2 experiments now drained.
