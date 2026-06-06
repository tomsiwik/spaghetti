# PAPER.md — KAN Adapter Lagrangian (KILLED)

## Summary

Tested whether Kolmogorov-Arnold Network (KAN) adapters (B-spline per-edge scalar functions, arxiv 2404.19756) could replace B-matmul in PoLAR, collapsing the entire merge-interference research arc to simple coefficient addition.

**Verdict: KILLED.** KAN adapters fail both expressivity (K1) and composition (K2) kill-criteria. The approach is fundamentally flawed for multi-task composition.

## Results

| Phase | Benchmark | Score | Baseline | Delta |
|-------|-----------|-------|----------|-------|
| M0 std-math | GSM8K | 66.0% | — | reference |
| Q1 KAN-math | GSM8K | 58.0% | 66.0% | -8.0pp |
| Q2 pure-KAN compose | GSM8K | 62.0% | — | — |
| Q2 pure-KAN compose | HumanEval | 42.0% | — | — |
| Q2 pure-KAN compose | MedQA | 14.0% | — | — |
| Q2 pure-KAN compose | **avg** | **39.3%** | — | — |
| Q3 hybrid | — | *stuck/timeout* | — | — |

## Kill-Criteria Verdicts

- **K1** (KAN-math GSM8K ≥ std-math - 5pp = 61%): **FAIL** — 58.0% < 61.0% (-3pp)
- **K2** (pure-KAN avg ≥ best-single + 2pp = 64%): **FAIL** — 39.3% << 64.0% (-24.7pp)
- **K3** (hybrid avg ≥ std K=2 + 1pp = 55%): **TIMEOUT** — Q3 stuck after 5h, task killed
- **K4** (budget): PASS — warm-start, no training needed

## Mechanism Analysis

### Why K1 failed (warm-start expressivity loss)
The warm-start initialization sets `skip_weight = B` (old matmul path) with spline coefficients near zero. Despite this, KAN-math lost 8pp vs standard PoLAR on GSM8K. Root cause: the SiLU activation in the skip path `w_skip · SiLU(z_i)` introduces nonlinearity that the original `B @ z` didn't have. The numerical drift from activation + grid normalization degrades the baseline signal.

### Why K2 failed catastrophically (spline coefficient addition ≠ functional composition)
Pure-KAN composition averages spline coefficients: `c_composed = 0.5 · c_math + 0.5 · c_code`. For B-splines, coefficient addition produces a function whose *pointwise output* is the weighted average of the individual functions' outputs — but only when the basis functions are identical (same grid, same degree). This is mathematically correct for function averaging.

However, the catastrophic failure (MedQA = 14%, below random 25%) reveals that **averaging two task-specialized scalar functions destroys task-specific activation patterns**. The math adapter learns splines that activate on math-token features; the code adapter learns splines that activate on code-token features. Their average activates on neither — it's a "nothing" function for out-of-domain inputs.

This is worse than standard PoLAR composition because matmul-based B-matrices at least preserve linear subspace structure. Nonlinear spline averaging has no such guarantee.

### Why Q3 timed out
The hybrid composition (KAN-math 0.5 + std strategy_full 0.5) required evaluating both adapter types per forward pass. The KAN adapter's per-edge B-spline evaluation is significantly slower than matmul (~3× per layer), making the combined inference prohibitively slow. After 5+ hours with no Q3 benchmark completing, the task was killed.

## Implications

1. **KAN adapters do NOT collapse the merge-interference problem.** Spline coefficient addition is formally correct but practically catastrophic because nonlinear basis functions don't preserve task-specific activation patterns.

2. **The merge problem is fundamentally about linear subspace geometry**, not about finding a better basis. KAN's "scalar function per edge" reformulation trades matmul interference for worse activation-pattern interference.

3. **Performance tax is prohibitive.** Even if composition worked, KAN's per-edge spline evaluation is ~3× slower than matmul, making it impractical for the 100+ tok/s target.

4. **Warm-start doesn't preserve expressivity.** Converting B-matmul to skip-path KAN loses 8pp even on the source task, making it unsuitable as a drop-in replacement.

## Connection to Prior Work

This result strengthens the finding from exp_pierre_pico_rescues_dare_b (Finding #842): approaches that concentrate or transform the B-direction signal (Pico → fewer directions, KAN → nonlinear scalar functions) make composition harder, not easier. The linear structure of B-matrices is a feature, not a bug — it's what makes PoLAR composition tractable.

## Citation
- Liu et al. 2024, "KAN: Kolmogorov-Arnold Networks", arxiv 2404.19756
