# Model Collapse Detection: Mathematical Foundations (v2)

## Notation

| Symbol | Shape / Domain | Definition |
|--------|---------------|------------|
| V | scalar, positive int | Vocabulary size (100 in experiment) |
| L | scalar, positive int | Sequence length (16 in experiment) |
| N | scalar, positive int | Number of sequences per cycle (200) |
| T | scalar, non-neg int | Self-learning cycle index |
| d | scalar, positive int | Model hidden dimension (64) |
| r | scalar, positive int | LoRA rank (4, 8, 16, 32, 64) |
| p_t | vector in R^V, simplex | Token distribution at cycle t |
| A | matrix in R^{r x d} | LoRA down-projection |
| B | matrix in R^{V x r} | LoRA up-projection |
| gamma | scalar in [0,1] | Zipf exponent for base distribution |
| tau | scalar > 0 | Sampling temperature |
| eta | scalar > 0 | Learning rate |

## Base Distribution Model

The base token distribution follows a Zipf law:

    p_base(i) = (1/i^gamma) / Z

where Z = sum_{i=1}^{V} 1/i^gamma is the normalization constant.

This gives logits:

    ell_base(i) = log p_base(i) = -gamma * log(i) - log(Z)

## LoRA-Constrained Expert Distribution

The expert distribution is a low-rank perturbation of the base:

    ell_expert(i) = ell_base(i) + [B @ (A @ h)]_i

where h in R^d is a context embedding. The key property: the perturbation
delta = B @ (A @ h) lies in the column space of B, which has rank at most r.

This means the expert distribution can only deviate from the base in at most
r independent directions in R^V. This is the rank bottleneck.

## Perturbation Norm Bound (FIX 5 -- corrected)

**Worst-case bound.** For the perturbation delta = B @ A @ h:

    ||delta||_2 = ||B @ A @ h||_2
               <= ||B||_op * ||A||_op * ||h||_2
               <= ||B||_F * ||A||_F * ||h||_2

where ||.||_op is the operator (spectral) norm and the second inequality
follows from ||M||_op <= ||M||_F for any matrix M.

Note: the v1 MATH.md claimed ||BA h||_2 <= ||B||_F * ||A||_F * ||h|| / sqrt(r).
This is INCORRECT as a worst-case bound. The sqrt(r) factor appears only in an
average-case argument: if the singular values of BA are roughly equal, each is
approximately ||BA||_F / sqrt(r). But this is not a valid upper bound. The
correct worst-case bound has no sqrt(r) divisor.

**Average-case interpretation.** If the singular values sigma_1, ..., sigma_r of
BA are approximately equal (sigma_i approx ||BA||_F / sqrt(r)), then the
"typical" perturbation magnitude along any single direction is smaller by
sqrt(r). This is a heuristic, not a bound.

**KL divergence bound.** For softmax distributions, if logits are perturbed by
delta, the KL divergence satisfies (via the log-sum-exp Lipschitz property):

    KL(p_expert || p_base) <= ||delta||_inf^2 / (2 * tau^2)

For a rank-r perturbation with norm-bounded LoRA matrices:

    ||delta||_inf <= ||delta||_2 <= ||B||_F * ||A||_F * ||h||_2

With our norm caps alpha_A = alpha_B = 5.0, ||h|| ~ 1:

    ||delta||_inf <= 25.0
    KL <= 25^2 / (2 * 0.8^2) = 488

This is a loose bound (the actual KL is much smaller because the rank constraint
distributes the perturbation across r directions, not concentrating it). The
key insight is not the specific bound value but the comparison:

    LoRA: delta in rank-r subspace of R^V, r directions of update
    Full-rank: delta in R^V, V directions of update

## Self-Training Update Model

At each cycle t:

1. Sample N sequences from p_t: x^{(n)} ~ Cat(p_t), n=1..N
2. Compute empirical distribution: hat{p}(i) = count(i) / (N * L)
3. Compute empirical logits: hat{ell}(i) = log(hat{p}(i) + eps)
4. Update LoRA matrices via gradient descent:
   - grad_direction = hat{ell} - ell_expert
   - Project through rank-r bottleneck (SVD truncation)
   - A <- A + eta * grad_A + noise
   - B <- B + eta * grad_B + noise
   - (With norm constraint) Apply: ||A||_F <= alpha, ||B||_F <= alpha

### Full-Rank Update (Two Baselines)

**Unanchored (v1 baseline):**

    ell_{t+1} = ell_t + eta * (hat{ell}_t - ell_t) + noise

All V dimensions are free to update. No rank bottleneck, no base anchoring.

**Anchored (v2 baseline, FIX 1):**

    delta_{t+1} = delta_t + eta * (hat{ell}_t - (ell_base + delta_t)) + noise
    ell_{t+1} = ell_base + delta_{t+1}

All V dimensions are free to update. No rank bottleneck, but anchored to base.
This isolates the rank effect from the base-anchoring effect.

## Why LoRA Regularizes Against Collapse (Revised Analysis)

### Finding: Rank constraint PLUS norm bounding prevents collapse

The v2 experiment reveals that TWO mechanisms jointly prevent collapse:

**Mechanism 1: Rank constraint (necessary but not sufficient).**
The rank bottleneck confines updates to a rank-r subspace of R^V.
This prevents the positive feedback loop from operating in all V directions
simultaneously. However, without norm bounding, the perturbation magnitude
in those r directions can grow without bound, eventually overwhelming the
base distribution and causing chaotic instability (not classical "collapse"
toward a spike, but divergence).

**Mechanism 2: Norm bounding (necessary but not sufficient).**
Capping ||A||_F and ||B||_F limits the total perturbation magnitude.
However, without rank constraint, a norm-bounded full-rank perturbation
can still cause collapse because the perturbation operates in all V directions,
and the positive feedback loop concentrates probability mass.

**Joint mechanism: rank + norm bounding (sufficient).**
Rank confines the perturbation to r directions. Norm bounding limits the
magnitude in those directions. Together, they prevent both collapse
(concentration) and instability (divergence).

### The Collapse Mechanism in Full-Rank (Anchored and Unanchored)

In full-rank self-training:
1. Generate from p_t (finite sample introduces noise)
2. Empirical distribution hat{p} concentrates on high-probability tokens
   (due to finite N -- tail tokens are undersampled)
3. Update moves p_{t+1} toward hat{p}, amplifying peaks
4. This is a positive feedback loop: peaked -> more peaked -> collapse

Base anchoring (computing ell = ell_base + delta rather than updating ell
directly) does NOT prevent this: the delta grows freely and overwhelms the
base. Experiment shows identical 73.0% diversity drop for both anchored
and unanchored full-rank at 5 cycles.

In LoRA + norm-bounded self-training:
- Updates are confined to r directions (rank constraint)
- Total perturbation magnitude is bounded (norm constraint)
- The feedback loop cannot simultaneously amplify all V tokens
- Distribution oscillates around the base rather than collapsing

## Diversity Metrics

### Unique n-gram ratio

    D_n(S) = |{(x_{i,j}, ..., x_{i,j+n-1}) : i=1..N, j=1..L-n+1}| / (N * (L-n+1))

### Embedding variance

    V_embed = trace(Cov(E[x^{(n)}]))

### Distribution entropy

    H(p) = -sum_{i=1}^{V} p(i) log_2 p(i)

### KL divergence from initial

    KL(p_t || p_0) = sum_{i=1}^{V} p_t(i) log(p_t(i) / p_0(i))

## Worked Example (Micro Scale, v2)

Parameters: V=100, L=16, N=200, d=64, r=16, eta=0.3

**Cycle 0 (LoRA, normed):**
- Base entropy: ~4.65 bits (Zipf with gamma=1.2)
- LoRA perturbation: ||delta||_2 <= ||B||_F * ||A||_F * ||h||_2 ~ 1.0 * 1.0 * 1.0 = 1.0
  (worst case; typical magnitude is smaller)
- Expert entropy: ~4.65 bits (small perturbation)
- n-gram diversity: ~18.7% unique bigrams

**After 5 cycles (LoRA r=16, normed):**
- Entropy drop: ~0.2% (essentially unchanged)
- n-gram diversity drop: ~1.6% (noise-level)
- No collapse detected

**After 5 cycles (LoRA r=16, NO norm constraint):**
- Entropy drop: ~15.6% (significant)
- n-gram diversity: chaotic (norms blow up, -217% "drop" = diversity increase from instability)
- Collapse rate: 90% of seeds

**After 5 cycles (anchored full-rank):**
- Entropy drop: ~40.5% (massive loss)
- n-gram diversity drop: ~73% (catastrophic collapse)
- 100% of seeds show collapse
- IDENTICAL to unanchored full-rank (anchoring provides no protection)

**Attribution:** Rank constraint + norm bounding jointly prevent collapse.
Neither alone is sufficient. Rank constraint is necessary to confine updates
to a low-dimensional subspace; norm bounding is necessary to prevent
divergence within that subspace.

## Assumptions

1. **Distribution-level model.** Real LLMs have complex conditional distributions
   per token position. We model a single unconditional token distribution.

2. **Weight normalization as regularizer.** We cap ||A||_F and ||B||_F to prevent
   unbounded growth. Real LoRA training uses weight decay (AdamW default: 0.01).
   The experiment shows this is CRITICAL, not incidental: without norm bounding,
   LoRA collapses at 80-100% of seeds.

3. **Random context embedding.** We use a fixed random context vector h.

4. **Zipf base distribution.** Reasonable approximation for natural language.

5. **Custom update rule (not standard LoRA SGD).** The gradient computation uses
   SVD projection rather than proper chain-rule backprop through BA. The key
   property preserved is that updates are confined to a low-rank subspace. The
   specific gradient dynamics differ from real LoRA training.

## Computational Cost

Per experimental condition: N_seeds * N_cycles * (sequence_generation + metric_computation)
- Total conditions: 5 ranks * 2 (normed/unnormed) + 2 baselines + 6 correlation + 4 fresh data + 1 detection = 23
- Runtime: ~120 seconds for full experiment suite
