# Composition Health via KL Divergence: Mathematical Foundations

## Variables and Notation

| Symbol | Shape | Description |
|--------|-------|-------------|
| W | (d, d') | Base model weight matrix (frozen) |
| delta_i | (d, d') | LoRA expert i's weight delta (= B_i @ A_i) |
| N | scalar | Number of composed experts |
| V | scalar | Vocabulary size (151,936 for Qwen2.5-7B) |
| T | scalar | Number of calibration tokens |
| x_t | (seq_len,) | Calibration text t (tokenized) |
| P_base(v \| x_t) | (V,) | Base model next-token distribution at last position of x_t |
| P_N(v \| x_t) | (V,) | Composed model (N experts) next-token distribution |
| P_{-i}(v \| x_t) | (V,) | Leave-one-out: composed without expert i |
| KL_N | scalar | Mean KL divergence at composition size N |
| DeltaKL_i | scalar | KL contribution of expert i |
| rho_K1 | scalar | Spearman correlation between DeltaKL and quality impact |

## Core Definitions

### KL Divergence Between Composed and Base

For a single calibration text x_t, the KL divergence from base to composed is:

```
KL(P_N || P_base)(x_t) = sum_v P_N(v | x_t) * log(P_N(v | x_t) / P_base(v | x_t))
```

where the sum is over all vocabulary tokens v. This measures how much the
composed model's predictions diverge from the base model's predictions.

**Direction convention:** KL(P_composed || P_base) measures how surprised
the base model would be by composed model outputs. This is the natural
direction: we want to detect when composition pushes predictions away
from the base's learned distribution.

The mean KL across T calibration texts:

```
KL_N = (1/T) * sum_{t=1}^{T} KL(P_N || P_base)(x_t)
```

### Per-Expert KL Contribution (Leave-One-Out)

For expert i in a composition of N experts:

```
DeltaKL_i = KL(P_N || P_base) - KL(P_{-i} || P_base)
```

where P_{-i} is the distribution of the model composed with all experts
except i.

**Interpretation:**
- DeltaKL_i > 0: Expert i increases divergence from base (additive KL contributor)
- DeltaKL_i < 0: Expert i actually pulls the composition closer to base (stabilizer)
- DeltaKL_i >> mean(DeltaKL): Expert i is an outlier -- potentially harmful

### Quality Impact (Ground Truth)

The "quality impact" of expert i is defined via its domain-specific PPL:

```
QualityImpact_i = PPL(W_base + sum_j delta_j, D_i) / PPL(W_base, D_i)
```

where D_i is expert i's evaluation data. Values < 1.0 mean the expert
helps on its own domain.

For cross-expert impact (does expert i hurt OTHER experts' domains):

```
CrossImpact_{i->j} = PPL(P_N, D_j) / PPL(P_{-i}, D_j)
```

Values > 1.0 mean expert i degrades expert j's domain.

## Theoretical Analysis

### Why KL Divergence Should Work

Consider the base model as defining a manifold of "natural" language distributions.
Each expert delta_i pushes the output distribution in a direction determined by
its learned specialization. For a well-trained expert:

```
P_expert_i(v | x) ~= P_base(v | x)  for x outside expert i's domain
P_expert_i(v | x) != P_base(v | x)  for x in expert i's domain
```

When we compose N experts additively:

```
logits_N = W_base @ h + sum_{i=1}^{N} delta_i @ h
```

The KL divergence on domain-agnostic calibration tokens measures the
"off-domain drift": how much the sum of all expert deltas perturbs the
base model's behavior on generic text.

### Decomposition of KL

For small perturbations (Taylor expansion around base logits):

```
KL(P_N || P_base) ~= (1/2) * h^T * (sum_i delta_i)^T * F * (sum_i delta_i) * h
```

where F is the Fisher information matrix of the base model at the calibration
point. This is a quadratic form in the total perturbation.

**Key insight:** This scales as O(||sum_i delta_i||^2_F), where F-norm is
weighted by Fisher information. At SOLE production cosines (~0.0002),
the cross terms are negligible:

```
||sum_i delta_i||^2_F ~= sum_i ||delta_i||^2_F + 2 * sum_{i<j} <delta_i, delta_j>_F
                       ~= sum_i ||delta_i||^2_F    (cross terms ~ 0)
```

So KL_N should grow approximately linearly with N when experts are orthogonal.
**Superlinear growth signals interference.**

### Expected KL Scaling

Under perfect orthogonality:

```
KL_N ~= c * N     (linear regime, healthy)
```

Under constructive interference (similar experts reinforce):

```
KL_N < c * N      (sublinear, experts partially cancel)
```

Under harmful interference:

```
KL_N >> c * N     (superlinear, experts amplify each other's off-domain effects)
```

The ratio KL_N / (c * N) is a composition health score. Values near 1.0 are
healthy; values >> 1.0 signal harmful interference.

### DeltaKL as Outlier Detector

For the leave-one-out statistic DeltaKL_i, under the null hypothesis that all
experts contribute equally:

```
E[DeltaKL_i] = KL_N / N   (each expert contributes 1/Nth of total KL)
Var[DeltaKL_i] depends on pairwise cosines and expert norms
```

A harmful expert produces DeltaKL_i >> E[DeltaKL_i]. We detect this via
z-score:

```
z_i = (DeltaKL_i - mean(DeltaKL)) / std(DeltaKL)
```

Flag expert i as suspicious if z_i > 2.0.

## Calibration Token Selection

The calibration set must be:

1. **Domain-agnostic:** No token should favor any specific expert's domain
2. **Diverse:** Cover multiple text types to detect broad distributional shift
3. **Short:** 100 tokens total (20 texts * 5 tokens each) for speed
4. **Fixed:** Same calibration set used for all compositions (enables comparison)

Categories for T=20 calibration texts:
- 4 Wikipedia-style factual (general knowledge)
- 4 code snippets (Python, SQL, bash, generic)
- 4 mathematical/scientific statements
- 4 conversational/QA-style prompts
- 4 creative/literary passages

Each text produces logits at its last token position, yielding T=20
calibration points for KL computation.

## Computational Complexity

### Single KL Measurement

```
Cost(KL_measurement) = T * C_forward
```

where C_forward is one forward pass through the model. For Qwen2.5-7B in
4-bit with T=20 texts of ~50 tokens each:

```
C_forward ~= 0.05s on A5000 (single text, 50 tokens)
Cost(KL_measurement) = 20 * 0.05 = 1.0s
```

K3 threshold is 30s per composition. With overhead (adapter loading, softmax
computation), expect 2-5s total. Well within budget.

### Full Leave-One-Out Protocol

For N experts, the leave-one-out requires N+1 KL measurements
(one full composition + N leave-one-out compositions):

```
Cost(full_LOO) = (N + 1) * Cost(compose_and_measure)
```

**Critical optimization:** Instead of reloading the base model N+1 times,
we can:

1. Pre-compute base logits once (reusable for all compositions)
2. Compose all N experts via weight-space merge (PEFT add_weighted_adapter)
3. For leave-one-out: subtract one expert's delta and re-forward

But PEFT does not support efficient single-expert removal. The practical
approach is:

**Option A (recompose each time):** N+1 full compositions. At ~10s per
composition (load base + merge + forward), this is ~510s for N=50.
Expensive but correct.

**Option B (pre-merge + delta subtraction):** Merge all N experts into
the base weights once. For leave-one-out of expert i, subtract delta_i's
contribution. This requires:
- One full merge (~10s)
- N forward passes with weight modification (~N * 2s)
Total: 10 + 50*2 = 110s for N=50. Feasible.

**Option C (batched base logits, incremental merge):** Compute base logits
once. For each N in [5, 10, 25, 50], compose incrementally. For leave-one-out,
only do it at N=10 (sufficient to test K2). Total:
- 1 base model load + 1 base logit computation: ~15s
- 4 compositions (N=5,10,25,50): ~40s
- 10 leave-one-out compositions: ~100s
- Total: ~155s (2.5 min)

**Recommended: Option C.** Keeps total runtime under 5 minutes. Sufficient
statistical power for all three kill criteria.

## Kill Criteria Mathematics

### K1: KL-Quality Correlation (Spearman rho >= 0.3)

We need to correlate two ranked lists:
1. DeltaKL_i for each expert (KL contribution)
2. QualityImpact_i for each expert (PPL ratio on its domain)

Spearman rho measures rank correlation. With N=10 leave-one-out samples:

```
Minimum detectable effect: rho >= 0.3 requires rho_true > 0.6 at power 0.8
                           (N=10 is low power; this is a directional test)
```

The test is inherently directional: if KL and quality loss are correlated
at rho > 0.3 at N=10, the mechanism works. If rho < 0.3, we need more
data or the mechanism is fundamentally flawed.

**Nuance:** The existing pilot50 benchmark gives per-adapter PPL on
contaminated eval data. We should use this as the quality signal despite
contamination, because KL is measured on different (calibration) tokens.
The PPL numbers are still a valid ranking of adapter quality even if
absolute values are inflated.

### K2: Harmful Expert Discrimination

To test K2, we need at least one "harmful" expert. Strategies:

**Natural approach:** Among 50 pilot adapters, find the one with worst
quality (highest PPL ratio). If DeltaKL for this adapter is distinguishable
from the mean, K2 passes.

**Synthetic approach:** Create a deliberately harmful adapter by:
- Random weight perturbation (pure noise delta)
- Or: flip the sign of a real adapter's weights (anti-expert)

The z-score test: z_harmful > 2.0 means the harmful expert's DeltaKL is
more than 2 standard deviations above mean. At N=10, this is detectable
with 95% confidence if the true effect is > 1 std above mean.

### K3: Speed (< 30s per composition)

KL measurement on 20 calibration texts requires 20 forward passes.
At ~0.05s per forward pass on A5000 with 4-bit Qwen2.5-7B:

```
Expected time = 20 * 0.05 = 1.0s (forward only)
                + 2.0s (softmax, KL computation)
                + 5.0s (adapter loading overhead)
                = ~8s per composition
```

This is 3.75x below the 30s threshold. K3 should pass easily.

## Worked Example (Micro-Scale Projection)

Consider d=64, r=8, N=8, T=20 calibration texts:

1. Base model computes logits on 20 texts: P_base(v | x_t) for t=1..20
2. Compose 8 experts: W_composed = W + sum_{i=1}^{8} B_i @ A_i
3. Compute P_8(v | x_t) on same 20 texts
4. KL_8 = mean over t of KL(P_8 || P_base)(x_t)

Expected values (based on micro findings):
- At N=50 pilot, catastrophic PPL (trillions) was observed, so KL_50 should
  be extremely large (hundreds to thousands of nats)
- At N=5, composition typically preserves quality, so KL_5 should be small
  (< 1 nat)
- The N=50 catastrophe suggests nonlinear KL growth, confirming the
  superlinear interference hypothesis

For leave-one-out at N=10:
- Expert with best quality (e.g., abstract-math, 28.87% PPL improvement):
  DeltaKL_math ~ small (good expert, small perturbation)
- Expert with worst quality (if any have negative improvement):
  DeltaKL_bad ~ large (harmful expert, large perturbation)
- z_score = (DeltaKL_bad - mean) / std > 2.0 for discrimination

## Assumptions

1. **Additive composition is the deployed method.** We test naive addition
   (W + sum delta_i), not Frechet merge (killed) or GS reorthogonalization.

2. **Calibration tokens are domain-agnostic.** The 20 texts are chosen to
   not favor any specific adapter's domain. This is verifiable by checking
   that base model PPL on calibration texts is similar across domains.

3. **KL is measured on logits, not probabilities.** We convert logits to
   log-probabilities via log_softmax, then compute KL. This avoids
   numerical issues with very small probabilities.

4. **4-bit quantization does not distort KL measurement.** The quantization
   affects absolute logit values but should preserve relative rankings.
   Both base and composed models use the same quantization config.

5. **Leave-one-out at N=10 has sufficient power.** N=10 gives limited
   statistical power for Spearman correlation. A rho of 0.3 at N=10
   corresponds to a p-value of ~0.4 (not significant). We treat this
   as a directional signal: if rho > 0.5, strong evidence; if rho < 0.0,
   mechanism is broken.

6. **The pilot50 PPL improvement rankings are valid.** Despite being
   measured on contaminated eval data, the relative ordering of adapter
   quality should be preserved. This is sufficient for rank correlation.
