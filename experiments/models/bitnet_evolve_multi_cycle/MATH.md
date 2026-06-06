# Multi-Cycle Evolve Convergence: Mathematical Foundations

## Setup

- Base model: BitNet-b1.58-2B-4T with ternary weights, dimension d = 2560
- LoRA adapters: rank r = 16, applied to all attention + MLP projections
- Domains: D = {medical, code} (two domains with established training data)
- Cycles: C = 3 (retrain-from-scratch per domain)
- Training budget per cycle: T = min(3 * |data|, 1000) steps

## Notation

| Symbol | Definition | Shape/Type |
|--------|-----------|------------|
| W_base | Frozen ternary base weights | R^{d_out x d_in} |
| A_i^{(c)} | LoRA A-matrix for domain i, cycle c | R^{d_in x r} |
| B_i^{(c)} | LoRA B-matrix for domain i, cycle c | R^{r x d_out} |
| Delta W_i^{(c)} | Adapter update: B_i^{(c)} @ A_i^{(c)} | R^{d_out x d_in} |
| PPL_i^{(c)} | Validation perplexity for domain i, cycle c | R+ |
| KR_i^{(c)} | KR-Test score for domain i, cycle c | [0, 1] |
| cos_{ij}^{(c)} | Cross-adapter cosine: |cos(Delta W_i^{(c)}, Delta W_j)| | [0, 1] |

## Hypothesis (Formal)

For each domain i in D, the sequence of validation perplexities across cycles
is monotonically non-increasing:

    PPL_i^{(1)} >= PPL_i^{(2)} >= PPL_i^{(3)}

and KR-Test scores are non-regressing:

    KR_i^{(c+1)} >= KR_i^{(c)} for all c in {1, 2}

## Why Monotonic Improvement Is Expected

### Mechanism: Quality-Gated Selection

Each cycle c produces a fresh adapter trained from random initialization.
The quality gate selects the adapter only if:

1. PPL_i^{(c)} < PPL_i^{(c-1)}  (strict PPL improvement)
2. KR_i^{(c)} - KR_base >= 0     (KR non-regression vs base)
3. max_j |cos_{ij}^{(c)}| < 0.05 (composition safety)

If the gate rejects, the previous cycle's adapter is retained. Therefore:

    PPL_best_i^{(c)} = min(PPL_i^{(1)}, ..., PPL_i^{(c)})

This is monotonically non-increasing by construction. The interesting question
is whether the TRAINING itself produces monotonic improvement, or whether the
gate must reject some cycles.

### Source of Variance Across Cycles

Even with identical data, fresh LoRA initialization introduces variance:
- A_i^{(c)} ~ Uniform(-1/sqrt(d_in), 1/sqrt(d_in))  (random)
- B_i^{(c)} = 0  (deterministic)

The loss landscape is non-convex, so different A initializations may converge
to different local optima. However, the predecessor experiment showed:
- Identical results across 3 seeds (CV ~ 0%)
- This suggests the loss landscape has a single dominant basin at this scale

### Progressive Data Improvement

Unlike the predecessor (which used the same 300 samples regardless of round),
this experiment uses the FULL dataset per cycle:
- Cycle 1: 500 samples, T = min(1500, 1000) = 1000 steps (2 epochs)
- Cycle 2: 500 samples, T = 1000 steps (2 epochs, different random seed)
- Cycle 3: 500 samples, T = 1000 steps (2 epochs, different random seed)

With identical data but different seeds, improvement comes from:
1. Exploring different regions of the loss landscape
2. Keeping only the best adapter (quality gate)

## Convergence Analysis

### Single Adapter Loss Bound

For a single-domain adapter trained with SGD for T steps at learning rate eta:

    E[L(theta_T)] <= L(theta*) + (||theta_0 - theta*||^2)/(2 * eta * T) + eta * sigma^2 / 2

where sigma^2 is the gradient noise variance. At T = 1000, this converges to
within O(1/T) of the optimum for convex sub-problems.

### Multi-Cycle Selection Bound

With C independent cycles, the best-of-C selection gives:

    E[min(PPL^{(1)}, ..., PPL^{(C)})] <= E[PPL^{(1)}] - sigma_PPL * Phi^{-1}((C-1)/C)

For C = 3 with Gaussian approximation:
    Improvement ~ sigma_PPL * 1.09

This is small when sigma_PPL is small (as the predecessor showed ~0% CV).
Therefore, the main source of improvement must come from the training itself,
not just selection.

## Composition Safety

Cross-adapter cosine similarity under Grassmannian A-matrices:

    |cos(Delta W_i, Delta W_j)| <= (alpha/r)^2 * ||B_i|| * ||A_i^T A_j|| * ||B_j||

With random A (not Grassmannian), the expected cosine is:

    E[|cos|] ~ sqrt(r / d_in) = sqrt(16 / 2560) = 0.079

The predecessor measured max |cos| ~ 0.014-0.016, well below 0.05. With
random A initialization, we expect similar values due to concentration of
measure in high dimensions.

## Kill Criteria (Formal)

**K1**: PPL plateaus or regresses by cycle 3.
- Kill if: PPL_i^{(3)} >= PPL_i^{(1)} for ANY domain i
- Equivalently: the training itself (not just the gate) must produce improvement

**K2**: KR-Test regresses.
- Kill if: KR_i^{(3)} < KR_i^{(1)} - epsilon for ANY domain i (epsilon = 0.02)

**K3**: Composition safety violated.
- Kill if: max |cos| > 0.05 for any cross-domain pair at any cycle

## Worked Example (Micro Scale)

- Domain: medical, d = 2560, r = 16
- Base PPL on medical val: ~15 (expected from prior experiments)
- Cycle 1: Train 1000 steps on 500 medical samples
  - Expected PPL: ~8-12 (50-80% of base, based on prior 2x improvement)
  - Expected KR: ~0.70-0.80 (medical had good contrastive signal)
- Cycle 2: Fresh init, same data, different seed
  - Expected PPL: ~8-12 (similar, small variance)
  - Quality gate: keep if PPL < cycle 1 PPL
- Cycle 3: Fresh init, same data, different seed
  - Expected PPL: ~8-12 (similar)
  - Quality gate: keep if PPL < best of cycles 1-2

Expected outcome: modest improvement (~5-15% PPL reduction over 3 cycles)
due to best-of-3 selection, with PPL trajectory mostly flat within noise.

## Computational Cost

Per cycle per domain:
- Training: 1000 steps * ~0.15s/step = 150s
- KR-Test: 50 pairs * ~2s/pair = 100s
- Val PPL: 25 batches * ~0.5s = 12s
- Total per cycle: ~260s

Total experiment:
- 2 domains * 3 cycles * 260s = ~26 min
- Plus model loading, data prep: ~5 min
- Total: ~31 min (within budget)
