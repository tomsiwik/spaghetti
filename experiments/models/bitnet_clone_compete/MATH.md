# Clone-Compete Evolution: Mathematical Foundations

## 1. Problem Setup

Given a set of N trained LoRA adapters {A_1, ..., A_N} on a frozen base model W,
where each adapter A_i = B_i @ A_i^T (rank-r factorization, B_i in R^{d x r}, A_i in R^{r x d}),
we want to **improve the worst-performing adapter** without regressing others.

**Definitions:**
- W: frozen ternary base weights (BitNet-2B-4T, d=2560)
- A_i: adapter i with parameters theta_i = {B_i, A_i} (rank r=16)
- PPL_i(D): perplexity of adapter i on domain D's validation set
- PPL_base(D): base model perplexity on domain D (no adapter)
- improvement_i = (PPL_base(D_i) - PPL_i(D_i)) / PPL_base(D_i)

**Target:** adapter A_k where k = argmin_i improvement_i (worst relative improvement)

## 2. Clone-Compete Protocol

### 2.1 Cloning

Clone A_k by copying its parameters:
```
theta_k' = theta_k  (deep copy)
```

### 2.2 Mutation via Continued Training

Fine-tune the clone on fresh data D_k' (disjoint from original training data D_k):
```
theta_k'^{(t+1)} = theta_k'^{(t)} - eta * grad_theta L(W + theta_k'^{(t)}, D_k'^{(t)})
```

where L is the cross-entropy loss. The clone starts from the original's parameters
(warm start), giving it a head start over training from scratch.

**Key insight:** Continued training on fresh data is equivalent to a **mutation** in
evolutionary terms. The clone inherits the original's learned features and adds
new knowledge from the unseen data.

### 2.3 Tournament Selection

Compare original A_k and clone A_k' on held-out data {x_1, ..., x_M}:

For each sample x_j:
```
PPL_orig(x_j) = exp(L(W + theta_k, x_j))
PPL_clone(x_j) = exp(L(W + theta_k', x_j))
win_j = 1 if PPL_clone(x_j) < PPL_orig(x_j), 0 otherwise
```

The tournament outcome is a Bernoulli sequence. Under H0 (no difference):
```
sum(win_j) ~ Binomial(n, 0.5)
```

where n = number of non-tied samples.

### 2.4 Statistical Test

Use binomial test for significance. For n decisive samples with k clone wins:
```
p-value = P(X >= k | X ~ Bin(n, 0.5))  (two-sided)
```

**Sample size requirement:** For effect size delta = p - 0.5 = 0.10 (clone wins
60% of decisive samples), power 0.8:
```
n >= (z_alpha/2 + z_beta)^2 / (4 * delta^2)
n >= (1.96 + 0.84)^2 / (4 * 0.01) = 196
```

So approximately 200 decisive samples needed for a reliable test at alpha=0.05.

**Sequential testing:** Can test after every batch of 10 samples and stop early
if p < alpha, with Bonferroni correction for multiple looks.

## 3. Regression Bound

When replacing A_k with A_k' in the composition, the composed model changes:

**Original composition (1/N scaling):**
```
Delta_orig = (1/N) * sum_i theta_i
Delta_clone = (1/N) * (sum_{i!=k} theta_i + theta_k')
```

**Difference:**
```
Delta_clone - Delta_orig = (1/N) * (theta_k' - theta_k)
```

The perturbation to any non-target domain j's output is:
```
||h_j^clone - h_j^orig|| = (1/N) * ||Delta_k' x_j - Delta_k x_j||
```

**Bound:** If ||theta_k' - theta_k||_F <= epsilon (clone stays close to original
in parameter space), then:
```
PPL regression on domain j <= O(epsilon / N)
```

This explains why K3 passes easily: the 1/N scaling suppresses cross-domain
interference from the evolved adapter.

## 4. Convergence of Iterated Evolution

After m rounds of clone-compete, the quality trajectory is:
```
PPL^{(0)} >= PPL^{(1)} >= PPL^{(2)} >= ... >= PPL^{(m)}
```

This monotonicity holds **given a sufficiently powered tournament** because:
1. The tournament selects the better adapter (selection pressure)
2. The clone starts from the winner (ratchet -- cannot regress given correct selection)
3. Fresh data provides new gradient signal (no diminishing returns from data reuse)

**Caveat:** The ratchet guarantee requires the tournament to correctly identify
the better adapter. With an underpowered tournament (e.g., p=0.265 at N=38),
the tournament may select a worse adapter with probability ~38%. Monotonicity
is guaranteed only in expectation when tournament power is sufficient.

**Diminishing returns:** As the adapter improves, the marginal benefit of fresh
data decreases. The improvement per round follows approximately:
```
PPL^{(t)} - PPL^{(t+1)} ~ C * (PPL^{(t)} - PPL_optimal)
```

where PPL_optimal is the best achievable PPL at this rank and data budget,
and C < 1 is the convergence rate.

## 5. Numerical Example (This Experiment)

- d = 2560, r = 16, N = 5 adapters
- Base: BitNet-2B-4T (ternary)
- Target: legal adapter (worst: improvement = 23.9% vs base 20.78)
- Clone training: 200 steps on fresh law-stack-exchange data

**Results:**
| Version | Legal PPL | Relative to Original |
|---------|-----------|---------------------|
| Original | 15.82 | -- |
| Clone v1 | 14.50 | -8.3% |
| Clone v2 | 13.04 | -17.6% |
| Base (no adapter) | 20.78 | +31.4% |

**Tournament v1:** 18 clone wins, 11 original wins, 9 ties (62.1% win rate)
- p-value = 0.265 (not significant at alpha=0.05 due to N=38)
- Expected: need ~200 samples for significance at delta=0.10

**Regression (1/N composition):**
- Max regression: +0.06% (math domain)
- All other domains improved slightly (negative delta)
- Well within 2% kill threshold

**Per-round improvement:**
- Round 1: -1.32 PPL (-8.3%)
- Round 2: -1.46 PPL (-10.1% from v1)
- Improvement persists across rounds (fresh data still helps; only 2 data points,
  insufficient to determine acceleration vs diminishing trend)

## 6. Computational Cost

- Clone training: 200 steps x 0.43s/step = 86s
- Tournament (38 samples, 2 evals each): ~30s
- Regression check (5 domains x 50 samples): ~60s
- Total per evolution round: ~3 minutes

At scale (N=100 adapters, 200-sample tournament):
- Training: 86s (same)
- Tournament: ~160s (200 samples x 2 evals x 0.4s)
- Regression: ~60s (only check non-target domains once)
- Total: ~5 minutes per evolution round

## 7. Assumptions

1. **Fresh data available:** Clone training requires unseen domain-specific data.
   At scale, this comes from HuggingFace datasets or continued user interactions.

2. **PPL correlates with quality:** At micro scale (2B), PPL is the only viable
   metric (task eval killed). At macro scale, supplement with task accuracy.

3. **Warm-start advantage:** Clone benefits from original's learned features.
   If original is very poor (near base), cold start might be equally good.

4. **1/N scaling:** Regression bound relies on 1/N composition. Without scaling,
   a large adapter change could cause catastrophic interference.

5. **Single seed:** Results from seed 42 only. Prior experiments show CV=0.5%
   across seeds for this architecture, partially mitigating.
