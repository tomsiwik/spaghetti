# Clone-Compete Powered Tournament: Mathematical Foundations

## 1. Problem Setup

Given N=5 LoRA adapters on BitNet-2B-4T (d=2560, r=16), we test whether
clone-compete evolution produces statistically significant improvement at
adequate sample size (N=200+), and whether the warm-start advantage is real.

**Arms:**
- A: original legal adapter (200 steps on law-stack-exchange)
- B: clone v2 (warm-started from A, 200+200 steps on legalbench)
- C: cold-start control (fresh LoRA from scratch, 400 steps on same legalbench data)

**Key property:** B and C have identical training data budgets (400 total steps
on legalbench) but different initialization: B inherits A's weights (warm start),
C starts from random LoRA initialization (cold start).

## 2. Sample Size Calculation

For a binomial test detecting delta = p - 0.5 = 0.05 (clone wins 55%) at
alpha=0.05, power=0.80:

```
n >= (z_{alpha/2} + z_beta)^2 / (4 * delta^2)
n >= (1.96 + 0.84)^2 / (4 * 0.0025) = 784
```

For delta = 0.10 (clone wins 60%): n >= 196.

We target 250 total samples expecting ~80% decisive (200 decisive).
Actual: 500 total, 419 decisive for clone-vs-original.

## 3. Statistical Tests

### 3.1 Binomial Test (Per-Sample Wins)

For n decisive samples with k wins for arm B:
```
H0: P(B wins) = 0.5
H1: P(B wins) != 0.5 (two-sided)
p = P(X >= k | X ~ Bin(n, 0.5)) + P(X <= n-k | X ~ Bin(n, 0.5))
```

### 3.2 Wilcoxon Signed-Rank (Paired Non-Parametric)

For paired PPL differences d_i = PPL_A(x_i) - PPL_B(x_i):
```
H0: median(d_i) = 0
Test statistic: W = sum(rank(|d_i|) * sign(d_i))
```

More powerful than binomial because it uses magnitude, not just direction.

### 3.3 Cohen's d (Effect Size)

```
d = mean(PPL_A - PPL_B) / SD(PPL_A - PPL_B)
```

Thresholds: |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.

## 4. The Win-Rate vs Aggregate Paradox

A critical finding: clone v2 LOSES per-sample (28.9%) but WINS on aggregate
(mean PPL diff = +0.63 in favor of clone). This occurs because:

```
mean(PPL_A - PPL_B) > 0  (clone better on average)
but
P(PPL_B < PPL_A) < 0.5   (clone loses more often)
```

This is mathematically possible when:
```
E[PPL_A - PPL_B | B wins] >> E[PPL_B - PPL_A | A wins]
```

i.e., when clone wins, it wins BIG (on hard samples). When original wins,
it wins small (on easy samples where both are good).

This is the heavy-tailed improvement pattern: clone concentrates its advantage
on a minority of high-difficulty samples.

## 5. Warm-Start vs Cold-Start Decomposition

The total improvement of clone v2 over original can be decomposed:

```
Delta_total = PPL_orig - PPL_clone = 15.82 - 13.04 = 2.78

This has two components:
Delta_data = PPL_orig - PPL_cold = 15.82 - 13.87 = 1.95  (additional data effect)
Delta_warm = PPL_cold - PPL_clone = 13.87 - 13.04 = 0.83  (warm-start advantage)
```

So: 70.1% of improvement is from additional data, 29.9% from warm-start.

**But on held-out tournament data** (distribution shift):
```
Aggregate PPL (tournament):
  original: 13.86
  clone_v2: 13.24
  cold_start: 13.23

Delta_total = 13.86 - 13.24 = 0.62
Delta_data = 13.86 - 13.23 = 0.63
Delta_warm = 13.23 - 13.24 = -0.01  (NEGATIVE: warm-start slightly hurts!)
```

On held-out data, 100% of the improvement is from data, 0% from warm-start.

## 6. Interpretation

The warm-start advantage observed on validation data (0.83 PPL) vanishes on
held-out tournament data (-0.01 PPL). This is consistent with the warm-start
overfitting to the original training distribution (law-stack-exchange), which
helps on the overlapping validation set but not on novel legal text.

The cold-start control, starting from random initialization, learns a more
generalizable representation from the legalbench data because it is not
biased by the original's distributional assumptions.

## 7. Assumptions

1. **PPL as quality metric.** Per-sample PPL comparison is the strongest
   available signal at 2B scale (task eval killed).

2. **Training budget equivalence.** Cold-start gets 400 steps to match
   clone's cumulative 400 steps (original 200 + clone 200). This is
   conservative: the original's 200 steps on law-stack-exchange data are
   "free" for the clone, giving it more diverse data exposure total.

3. **Seed determinism.** Cold-start uses the same random seed for LoRA
   initialization as all prior experiments (implicit in MLX defaults).
   Different seeds would produce slightly different cold-start adapters.

4. **Tournament data independence.** Tournament uses law-stack-exchange
   samples 500+ and lex_glue ecthr_a (European Court of Human Rights).
   These are distribution-shifted from both training sources.
