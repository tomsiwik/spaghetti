# Self-Routing Adapters: Mathematical Foundations

## Setup and Notation

| Symbol | Definition | Shape |
|--------|-----------|-------|
| h | Hidden state from base model's last layer | (d,) where d=2560 |
| A_i | Frozen Grassmannian projection for adapter i | (d_in, r) where r=16 |
| B_i | Learned adapter weights for adapter i | (r, d_out) |
| N | Number of adapters/domains | 49 |
| L | Number of transformer layers | 30 |
| r | LoRA rank | 16 |

## Candidate Routing Signals

### Method A: Quadratic Form (B-only)

For adapter i, the routing score is the energy of h in B_i's row space:

$$s_i^{(A)} = \|B_i \cdot h\|_2^2 = h^T B_i^T B_i h$$

where B_i: (r, d), so B_i^T B_i: (d, d) is rank-r.

**Complexity:** O(d*r) per adapter = O(2560 * 16) = 40,960 FLOPs.

**Problem:** B_i^T B_i projects onto a rank-16 subspace of R^2560.
In high dimensions, random subspaces are nearly orthogonal.
For N=49 adapters each with r=16, we're comparing 16-dimensional
subspaces in 2560-dimensional space. The expected overlap between
random subspaces is O(r^2/d) = O(256/2560) ~ 0.1, meaning all
projections capture roughly the same fraction of any given h.
This is the "concentration effect" -- confirmed empirically at ~2% accuracy.

### Method B: Full Activation Norm (AoE-style)

$$s_i^{(B)} = \|B_i \cdot A_i^T \cdot h\|_2$$

This is the norm of the adapter's full output delta_i(h) = B_i @ A_i^T @ h.

**Complexity:** O(d*r + r*d_out) per adapter. For q_proj:
O(2560*16 + 16*2560) = 81,920 FLOPs.

**Why it might work:** A_i is domain-agnostic (Grassmannian, frozen) but
B_i is domain-specific (learned). The composition A_i^T @ h first projects
to the rank-r subspace, then B_i maps to output. If B_i learned domain-specific
patterns, domains that "activated" this adapter during training should produce
larger output norms.

**Empirical result:** ~23.5% top-2 accuracy. Better than random (4.1%) but
far below the 86.3% baseline. The signal exists but is weak.

### Method C: SVD Projection

Compute SVD of B_i^T = U_i S_i V_i^T, take U_i[:, :r]:

$$s_i^{(C)} = \frac{\|U_i U_i^T h\|_2}{\|h\|_2}$$

This measures the fraction of h that lies in the column space of B_i^T.

**Empirical result:** ~2.7% top-2, essentially random. The SVD basis of B
is too generic to discriminate domains.

### Method D: Hidden-State Centroid (the winner)

For each adapter i, compute the centroid of training hidden states:

$$\mu_i = \frac{1}{|D_i|} \sum_{h \in D_i} h$$

Route via cosine similarity:

$$s_i^{(D)} = \frac{h \cdot \mu_i}{\|h\| \cdot \|\mu_i\|}$$

**Complexity:** O(N * d) per token = O(49 * 2560) = 125,440 FLOPs.
Pre-computation: store N centroids of size d = 49 * 2560 * 4B = 500KB (125K floats).

**Parameters:** This method has zero *learned* parameters but requires
125,440 *computed* parameters (the stored centroids) plus 20 labeled
examples per domain for centroid estimation. The honest comparison
with Gumbel-sigmoid (659K learned params) is: closed-form solution
from labeled data vs. gradient-optimized solution from labeled data.

**Why it works:** Hidden states from the same domain cluster tightly
in the base model's representation space. The base model has ALREADY
learned domain-discriminative features. The centroid captures this.
This is nearest-centroid classification (Prototypical Networks, Snell
et al. 2017) applied to adapter routing -- a well-established technique
when the pretrained encoder produces separable features.

## Why B-Matrix Routing Fails

### Theoretical intuition: Concentration of Measure

**Theorem (for random subspaces).** In R^d with d >> r, for a random
unit vector u and a random r-dimensional subspace S drawn uniformly
from the Grassmannian:

$$\mathbb{E}\left[\frac{\|P_S u\|^2}{\|u\|^2}\right] = \frac{r}{d}$$

with variance O(r/d^2). As d/r grows, ALL random vectors project with
nearly identical norm onto ANY random rank-r subspace.

**Important caveat:** This theorem requires both the subspace S and
vector u to be random (or at least S to be uniformly distributed on
the Grassmannian). The B-matrices are *trained* on domain-specific data
and are NOT random subspaces. A sufficient condition for B-matrix routing
to work would be if domains induced *structured* hidden states that
aligned differently with each adapter's learned subspace. The
concentration theorem does not rule this out in principle.

### Empirical observation: B-matrices behave like random subspaces

**Our setting:** d=2560, r=16, so r/d = 0.00625. The concentration
theorem predicts ~0.625% projection fraction with low variance for
random subspaces.

**Empirical result:** B-matrix routing achieves 2.0% top-1, 4.7% top-2
accuracy -- statistically indistinguishable from random chance (2.0%
top-1, 4.1% top-2). Despite B-matrices being domain-specific (inter-
adapter cosine ~0.03), their rank-16 subspaces empirically fail to
develop domain-discriminative structure in R^2560.

**Interpretation:** The trained B-matrices, despite capturing domain-
specific patterns in their learned weights, span subspaces that are too
small (r/d = 0.006) relative to the ambient dimension for their subspace
orientation to discriminate inputs. The concentration theorem correctly
predicts this behavior, but the evidence is empirical -- it is the
measured routing accuracy, not the theorem alone, that kills B-matrix
routing.

## Why Hidden-State Centroids Work

The base model (BitNet-2B-4T) was pretrained on diverse data. Its
hidden representations already encode domain information. The mean
hidden state per domain is a sufficient statistic for domain classification
because:

1. **Cluster separability:** Different domains occupy distinct regions
   of the hidden state space. At d=2560, there is ample room for 49
   well-separated clusters.

2. **Low-dimensional structure:** Despite d=2560, the effective
   dimensionality of domain clusters is much lower. Cosine similarity
   suffices for nearest-centroid classification.

3. **No training needed:** The centroids can be computed from a few
   examples per domain (we used 20). This is a k=1 nearest-centroid
   classifier in the base model's representation space.

## Worked Example (d=2560, N=49, r=16)

Given a hidden state h from the "code" domain:
- Raw centroid routing: cos(h, mu_code) = 0.89, cos(h, mu_math) = 0.72
  => correctly routes to "code" adapter
- B-matrix quadratic form: s_code = 1.247, s_chemistry = 1.253
  => incorrectly routes to "chemistry" (scores near-identical due to concentration)

## Computational Cost Comparison

| Method | FLOPs/token | Params | Top-2 Acc |
|--------|-------------|--------|-----------|
| Random | 0 | 0 | 4.1% |
| B-only quadratic | 40K | 0 learned | 4.7% |
| Full activation norm (1 layer) | 82K | 0 learned | 21.0% |
| Full activation norm (5 layers) | 410K | 0 learned | 23.5% |
| SVD projection | 82K + SVD | 0 learned | 2.7% |
| Centroid cosine (closed-form) | 125K | 125K computed | 87.1% |
| Gumbel-sigmoid (3000 steps) | 670K | 659K learned | 86.3% |
| Gumbel-sigmoid (6000 steps) | 670K | 659K learned | 90.4% |

## Assumptions

1. Hidden states are from the base model's last hidden layer (not
   from after adapter application). This is the standard routing point.
2. A matrices are frozen Grassmannian (verified).
3. B matrices were trained independently per domain on domain-specific data.
4. Evaluation uses 10 held-out samples per domain (490 total).
5. Domain labels are clean (each sample belongs to exactly one domain).
