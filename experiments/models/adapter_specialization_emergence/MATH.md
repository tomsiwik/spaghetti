# Mathematical Foundations: Adapter Specialization Emergence

## 1. Mechanism Definition

### Setup

We train N adapters on **identical** mixed-domain data. Each adapter i has:
- Frozen A_i in R^{d x r}, drawn from the Grassmannian skeleton (AP-packed, mutually orthogonal)
- Trainable B_i in R^{r x d_out}, initialized to zero, STE-ternarized during training

The weight update for adapter i at layer l is:
  delta_W_i = B_i @ A_i^T   (shape: d_out x d)

with LoRA applied as: y = W_base @ x + scale * (B_i @ (A_i^T @ x))

### The Key Constraint

All N adapters see the same training data D = {x_1, ..., x_T} (a uniform mixture of
K domains d_1, ..., d_K). The ONLY difference between adapters is their frozen A_i matrix.

Since A_i is frozen and orthogonal to A_j (i != j), the gradient for B_i at step t is:

  dL/dB_i = dL/dy @ x^T @ A_i / scale

The projection x^T @ A_i extracts different r-dimensional features from the input for
each adapter. This is the FlyLoRA mechanism (arxiv 2510.08396): frozen random A acts
as an implicit feature selector via the Johnson-Lindenstrauss lemma.

### What We Measure

Build an N x K PPL matrix P where P_{i,k} = PPL of adapter i on domain k's eval data.

Specialization is measured by:
1. **Per-adapter best domain**: argmin_k P_{i,k} -- does each adapter have a distinct "favorite"?
2. **Silhouette score**: Cluster adapters by their PPL profiles, measure separability
3. **Entropy of domain preferences**: H = -sum p_k log p_k where p_k = fraction of
   adapters whose best domain is k

## 2. Why It Should Work

### Gradient Decorrelation via Orthogonal Projections

For two adapters i,j with A_i^T A_j = 0, the gradient correlation is:

  E[dB_i^T dB_j] = E[(dL/dy x^T A_i)^T (dL/dy x^T A_j)]
                  = E[A_i^T x x^T A_j] * E[(dL/dy)^2]
                  = A_i^T Sigma_x A_j * C

where Sigma_x = E[xx^T] is the input covariance. If x is isotropic (Sigma_x = sigma^2 I),
then A_i^T Sigma_x A_j = sigma^2 A_i^T A_j = 0, so gradients are perfectly decorrelated.

In practice, data is NOT isotropic. The covariance has non-trivial structure, so
A_i^T Sigma_x A_j != 0 in general. However, the JL lemma guarantees that for
d >> r, random r-dimensional projections approximately preserve inner products:

  |A_i^T Sigma_x A_j| <= O(sqrt(r/d)) * ||Sigma_x||

At d=2560, r=16: sqrt(r/d) = sqrt(16/2560) = 0.079, so cross-correlations are
suppressed by ~13x relative to self-correlations.

### MoE Self-Specialization Analogy

In Mixture-of-Experts models (Shazeer et al., 2017; Fedus et al., 2022), experts
trained on mixed data self-specialize even without explicit domain labels. The mechanism:
experts that happen to be slightly better at certain inputs receive more gradient signal
for those inputs via the gating function.

Here, we have NO gating during training -- each adapter sees ALL data equally. The
specialization mechanism, if it exists, must come purely from the A-matrix projection
selecting different features from the input.

### FlyLoRA Evidence (arxiv 2510.08396)

FlyLoRA showed that frozen random A matrices act as implicit routers: different A
matrices cause different adapters to learn different aspects of the data, even when
trained on the same distribution. Their key finding: frozen A LoRA matches trained-A
LoRA quality, and the JL lemma provides theoretical grounding for why random projections
preserve enough structure.

## 3. What Breaks It

### Convergence to Same Optimum

If the loss landscape has a single global minimum in B-space that is independent of
the A-projection direction, all adapters will learn equivalent B matrices (up to
the A rotation). In this case, P_{i,k} will be approximately constant across i for
each k, and silhouette will be ~0.

**Kill condition (K1):** silhouette < 0.2 means random clustering -- adapters did NOT
specialize.

### Capacity Limitation

At rank r=16 and 200 training steps on mixed data (~500 samples), each adapter may
not have enough capacity or training signal to learn domain-specific features. The
effective capacity per domain is:

  params_per_domain = (r * d_out * n_layers * n_modules) / K

At r=16, d_out=2560, 30 layers, 7 modules, K=10 domains:
  = 16 * 2560 * 30 * 7 / 10 = 860,160 effective params/domain

This is generous. The concern is training steps: 200 steps on 500 samples means
each sample seen ~0.4 times on average, which may be insufficient for specialization.

### Domain Similarity

If the 10 selected domains are too similar in the model's feature space (as
softmax_router_scaling showed: 8/24 domains cluster together), orthogonal
projections may still capture overlapping structure, yielding weak specialization.

## 4. Connection to Architecture

This experiment tests a core question for SOLE: **can we replace explicit
domain-labeled training with mixed training + orthogonal-A-induced specialization?**

If YES: dramatically simplifies the training pipeline -- no need for per-domain
data curation. Just train N adapters on the same dump and let A matrices force
specialization.

If NO: confirms that explicit domain training is necessary, and the A matrices
serve only as interference prevention (already proven), not as specialization inducers.

Either result is informative for the SOLE architecture.

## 5. Complexity Analysis

- Training: N * 200 steps, each on full model. ~20s per adapter. Total: ~200s for N=10.
  (Adapters trained sequentially, each loads fresh model)
- Evaluation: N * K forward passes (25 samples each). ~5s per eval. Total: ~500s.
- Comparison with domain-trained: K forward passes. ~50s.
- Total runtime estimate: ~15-20 minutes.
- Memory: ~17GB peak (same as single adapter training on BitNet-2B).

## 6. Worked Example (Conceptual)

With N=3 adapters on K=3 domains (code, math, medical):

If specialization occurs, the PPL matrix looks like:
```
         code  math  medical
Adapter0  5.2   8.1    9.3     <- specializes code
Adapter1  7.8   4.9    8.7     <- specializes math
Adapter2  8.5   7.2    5.1     <- specializes medical
```

If NO specialization:
```
         code  math  medical
Adapter0  6.5   6.7    7.1
Adapter1  6.4   6.8    7.0
Adapter2  6.6   6.6    7.2
```

The silhouette score distinguishes these: ~0.5+ for specialized, ~0.0 for uniform.

## 7. Prior Art

- **FlyLoRA** (arxiv 2510.08396): Frozen random A as implicit feature selector. Shows
  JL-lemma guarantees on projection quality. Motivates our hypothesis.
- **MoE Self-Specialization** (Shazeer et al., 2017): Experts self-specialize on
  mixed data via gating gradients. Our setup removes gating -- purer test of
  projection-induced specialization.
- **exp_softmax_router_scaling LEARNINGS:** Showed 8/24 domains cluster in hidden
  space. Suggests specialization may be coarse-grained (cluster-level, not domain-level).
- **exp_real_data_25_domain_adapters:** Provides trained domain-specific baselines
  and infrastructure (data loading, Grassmannian skeleton, evaluation).
